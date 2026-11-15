"""Offline VOP training with validation after each epoch."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import signal
import time

import torch
from torch.utils.data import DataLoader

from ..config import ModelConfig, TrainConfig, parse_config, save_config
from .dataset import VOPDataset, ShardBatchSampler
from ..model import VOPNet, load_model
from ..metrics import vop_loss, batch_metrics
from ..progress import format_training_progress


def _atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def _loader(dataset, config, shuffle):
    sampler = ShardBatchSampler(dataset, config.batch_size, shuffle, config.seed)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=config.workers,
                      pin_memory=torch.device(config.device).type == 'cuda',
                      generator=torch.Generator().manual_seed(config.seed)), sampler


def run_epoch(model, loader, device, alpha=.2, optimizer=None, progress=None, check_stop=None):
    training = optimizer is not None
    model.train(training)
    totals, samples = {}, 0
    epoch_started = time.monotonic()
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for index, (history, target) in enumerate(loader):
            if check_stop:
                check_stop()
            history, target = history.to(device, non_blocking=True), target.to(device, non_blocking=True)
            prediction = model(history)
            loss, _ = vop_loss(prediction, target, alpha)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite VOP loss')
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
            metrics = batch_metrics(prediction.detach(), target, alpha)
            count = len(history)
            samples += count
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.) + value * count
            if progress and ((index + 1) % 100 == 0 or index + 1 == len(loader)):
                progress(index + 1, len(loader),
                         {name: value / samples for name, value in totals.items()} | {'samples': samples},
                         time.monotonic() - epoch_started)
    if not samples:
        raise ValueError('empty epoch')
    return {name: value / samples for name, value in totals.items()} | {'samples': samples}


def train(dataset, run_dir, config=None, model_config=None, check_stop=None):
    config = config or TrainConfig()
    torch.set_num_threads(config.num_threads)
    torch.manual_seed(config.seed)
    if torch.device(config.device).type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; select --device cpu for CPU training')
        torch.cuda.manual_seed_all(config.seed)
    train_data = VOPDataset(dataset, 'train', verify=True, check_stop=check_stop)
    val_data = VOPDataset(dataset, 'val', verify=True, check_stop=check_stop)
    maximum = train_data.manifest['label_config']['maximum_speed_mps']
    model_config = model_config or ModelConfig(maximum=maximum)
    if not math.isclose(model_config.maximum, maximum):
        raise ValueError('model maximum differs from label maximum')
    run_dir = Path(run_dir).resolve()
    if run_dir.exists() and any(p.name not in ('launch.log', 'train_vop.log', 'resolved.yaml') for p in run_dir.iterdir()):
        raise FileExistsError('Run folder must be new or empty')
    model = VOPNet(model_config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    start, best = 0, math.inf
    writer = None
    if config.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'checkpoints').mkdir(exist_ok=True)
    _atomic_json(run_dir / 'config.json', dict(model=asdict(model_config), training=asdict(config)))
    _atomic_json(run_dir / 'dataset_info.json', dict(fingerprint=train_data.fingerprint, **train_data.manifest))
    if config.tensorboard:
        writer = SummaryWriter(str(run_dir / 'tensorboard'))
    train_loader, sampler = _loader(train_data, config, True)
    val_loader, _ = _loader(val_data, config, False)
    started = time.monotonic()

    def log(message):
        print(message, flush=True)

    def report(phase):
        def callback(completed, total, values, phase_elapsed):
            log(format_training_progress(epoch, config.epochs, phase, completed, total, values,
                elapsed=time.monotonic() - started, phase_elapsed=phase_elapsed))
        return callback

    try:
        for epoch in range(start + 1, config.epochs + 1):
            if check_stop:
                check_stop()
            epoch_started = time.monotonic()
            sampler.epoch = epoch - 1
            _atomic_json(run_dir / 'state.json', dict(status='training', epoch=epoch, epochs=config.epochs))
            training = run_epoch(model, train_loader, config.device, config.alpha, optimizer, report('train'), check_stop)
            _atomic_json(run_dir / 'state.json', dict(status='validating', epoch=epoch, epochs=config.epochs))
            validation = run_epoch(model, val_loader, config.device, config.alpha, progress=report('val'), check_stop=check_stop)
            if check_stop:
                check_stop()
            if not all(math.isfinite(x) for row in (training, validation) for x in row.values()):
                raise FloatingPointError('nonfinite epoch metric')
            improved = validation['loss'] < best
            best = min(best, validation['loss'])
            checkpoint = dict(schema_version=1, epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                model_config=asdict(model_config), train_config=asdict(config), best_val_loss=best,
                dataset_fingerprint=train_data.fingerprint, input_config=train_data.manifest['input_config'],
                label_config=train_data.manifest['label_config'], torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.device(config.device).type == 'cuda' else None)
            for name in (['last.pt', 'best.pt'] if improved else ['last.pt']):
                path = run_dir / 'checkpoints' / name
                temporary = path.with_suffix('.pt.tmp')
                torch.save(checkpoint, temporary)
                temporary.replace(path)
            metrics = dict(epoch=epoch, train=training, val=validation, best_val_loss=best,
                           elapsed_s=time.monotonic() - started)
            with (run_dir / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(metrics) + '\n')
            if writer:
                for split, row in (('train', training), ('val', validation)):
                    for key, value in row.items():
                        writer.add_scalar(f'{split}/{key}', value, epoch)
                writer.flush()
            elapsed = time.monotonic() - started
            log(format_training_progress(epoch, config.epochs, 'completed', len(val_loader), len(val_loader),
                dict(train=training, val=validation, samples=training['samples'] + validation['samples']),
                elapsed=elapsed, phase_elapsed=time.monotonic() - epoch_started, best=best,
                eta=elapsed / (epoch - start) * (config.epochs - epoch)))
        _atomic_json(run_dir / 'state.json', dict(status='completed', **metrics))
        return metrics
    except BaseException as error:
        _atomic_json(run_dir / 'state.json', dict(status='interrupted' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'failed',
                                                 error=repr(error)))
        raise
    finally:
        if writer:
            writer.close()


def main():
    args = parse_config('train_vop')
    dataset, run_dir = args.dataset, args.run_dir
    if dataset is None:
        raise ValueError('Set dataset in the VOP training YAML file')
    if not (dataset / 'manifest.json').is_file() and (dataset / 'prepared').is_dir():
        dataset = dataset / 'prepared'
    maximum = json.loads((dataset / 'manifest.json').read_text())['label_config']['maximum_speed_mps']
    config = TrainConfig(**{name: getattr(args, name) for name in TrainConfig.__dataclass_fields__})
    model_config = ModelConfig(channels=args.channels, hidden=args.hidden, maximum=maximum)
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    def check_stop():
        if stopping or (run_dir / 'stop_requested.json').exists():
            raise InterruptedError('Training stop requested; only complete epochs are checkpointed')

    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        save_config(args, run_dir)
        train(dataset, run_dir, config, model_config, check_stop=check_stop)
        save_config(args, run_dir)
    except InterruptedError as error:
        run_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(run_dir / 'state.json', dict(status='interrupted', error=str(error)))
        print(str(error), flush=True)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
