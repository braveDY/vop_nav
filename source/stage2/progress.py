

def duration(seconds):
    hours, remainder = divmod(int(max(0, seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def print_progress(phase, completed, total, elapsed, fields):
    eta = elapsed / completed * max(0, total - completed)
    rows = [f'{name + ":":>28} {value}' for name, value in fields.items()]
    print('\n'.join([
        '#' * 80, f'{phase} file {completed}/{total}'.center(80), '', *rows,
        '-' * 80, f'{"Phase elapsed:":>28} {duration(elapsed)}',
        f'{"Phase ETA:":>28} {duration(eta)}', '',
    ]), flush=True)


def format_training_progress(epoch, epochs, phase, completed, total, metrics, *,
                             elapsed, phase_elapsed, best=None, eta=None):
    rows = []
    def add(name, value):
        rows.append(f'{name + ":":>35} {value}')
    add('Phase', f'{phase} batch {completed}/{total}')
    if phase_elapsed > 0:
        add('Computation', f"{metrics['samples'] / phase_elapsed:.0f} samples/s")
    groups = metrics if 'train' in metrics else {phase: metrics}
    for split, values in groups.items():
        if not isinstance(values, dict):
            continue
        for label, key in (('loss', 'loss'), ('IoU', 'iou'), ('MSE', 'mse'),
                           ('boundary MAE (m/s)', 'boundary_mae_mps'),
                           ('unsafe length (m/s)', 'unsafe_length_mps'),
                           ('missed safe length (m/s)', 'missed_safe_length_mps')):
            add(f'{split.capitalize()} {label}', f'{values[key]:.6f}')
        add(f'{split.capitalize()} samples', f"{values['samples']:,}")
    if best is not None:
        add('Best validation loss', f'{best:.6f}')
    rows.append('-' * 80)
    add('Phase time', duration(phase_elapsed))
    add('Time elapsed', duration(elapsed))
    add('ETA', duration(eta) if eta is not None else 'pending first complete epoch')
    return '\n'.join(['#' * 80, f' Learning epoch {epoch}/{epochs} '.center(80), '', *rows, ''])
