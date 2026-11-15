/* Plain scripts and per-environment data keep file:// replay completely offline. */
"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const manifest = window.VOP_REPLAY_MANIFEST;
  const cache = new Map();
  let data = null, wanted = null, frame = 0, start = 0, end = 0, playing = false;
  let clock = 0, lastTick = null;
  const canvas = $("sceneCanvas"), ctx = canvas.getContext("2d");
  const voView = window.VOPView.create(() => { pause(); render(); });
  let sceneProjection = null;
  const color = {robot:"#6dc9ec", moving:"#edac65", static:"#9da8b8", lidar:"#75d7a7", goal:"#e5bbef"};
  const controls = ["playButton","prevButton","nextButton","startButton","timeline"];

  function error(message) { $("error").textContent = message; $("error").hidden = false; }
  function pause() { playing = false; lastTick = null; $("playButton").textContent = "Play"; }
  function decode(field) {
    const bytes = Uint8Array.from(atob(field.data), (c) => c.charCodeAt(0));
    const types = {f32:Float32Array, u32:Uint32Array, u8:Uint8Array};
    if (!types[field.dtype]) throw new Error("Unsupported data type: " + field.dtype);
    const array = new types[field.dtype](bytes.buffer);
    if (field.shape.reduce((a,b) => a*b, 1) !== array.length) throw new Error("Data shape mismatch");
    return array;
  }

  window.VOP_REPLAY_LOAD = (id, payload) => {
    try {
      const decoded = Object.fromEntries(Object.entries(payload.fields).map(([key,value]) => [key,decode(value)]));
      cache.delete(id);
      cache.set(id, {...payload, fields:decoded});
      while (cache.size > 3) cache.delete(cache.keys().next().value);
      if (id === wanted) useData(cache.get(id));
    } catch (e) { if (id === wanted) error("Failed to load environment data: " + e.message); }
  };

  function chooseEnvironment(id, autoplay = false) {
    pause(); wanted = id; data = null;
    playing = autoplay;
    $("playButton").textContent = autoplay ? "Pause" : "Play";
    $("error").hidden = true;
    controls.forEach((key) => { $(key).disabled = true; });
    draw();
    if (cache.has(id)) { useData(cache.get(id)); return; }
    const entry = manifest.environments.find((item) => item.id === id);
    if (!entry) { error("Environment data not found"); return; }
    const script = document.createElement("script");
    script.src = entry.file;
    script.onload = () => script.remove();
    script.onerror = () => { script.remove(); if (id === wanted) error("Failed to load " + entry.file + ". Keep index.html and the data folder together."); };
    document.head.appendChild(script);
  }

  function useData(value) {
    data = value; frame = 0; clock = 0;
    controls.forEach((key) => { $(key).disabled = false; });
    start = 0; end = data.frames-1;
    $("timeline").min = start; $("timeline").max = end;
    setFrame(start);
  }

  function setFrame(value) {
    frame = Math.max(start, Math.min(end, Math.round(value)));
    clock = frame; $("timeline").value = frame;
    render();
  }

  function render() {
    if (!data) return;
    const f = data.fields, degrees = 180 / Math.PI;
    $("speedValue").textContent = Math.hypot(f.robot_velocity[2*frame], f.robot_velocity[2*frame+1]).toFixed(2);
    $("heightValue").textContent = f.robot_height[frame].toFixed(2);
    $("episodeValue").textContent = f.episode_id[frame];
    $("stepValue").textContent = f.episode_step[frame];
    $("yawValue").textContent = (f.robot_yaw[frame]*degrees).toFixed(1) + "°";
    $("attitudeValue").textContent = `${(f.robot_attitude[2*frame]*degrees).toFixed(1)}° / ${(f.robot_attitude[2*frame+1]*degrees).toFixed(1)}°`;
    $("historyValue").textContent = f.history_valid[frame] ? "Ready" : "Fewer than 5 frames";
    $("timeValue").textContent = (manifest.global_steps[frame]*manifest.dt).toFixed(2) + " s";
    $("durationValue").textContent = `${manifest.duration_s.toFixed(2)} s`;
    $("frameValue").textContent = `Frame ${frame} / ${data.frames-1}`;
    draw();
  }

  function draw() {
    voView.update(data, manifest, frame);
    const box = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1, width = Math.max(box.width,1), height = Math.max(box.height,1);
    if (canvas.width !== Math.round(width*dpr) || canvas.height !== Math.round(height*dpr)) {
      canvas.width = Math.round(width*dpr); canvas.height = Math.round(height*dpr);
    }
    ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,width,height);
    const bounds = voView.getBounds() ?? {minX:-5.8,maxX:5.8,minY:-3.4,maxY:3.4};
    const scale = Math.min((width-76)/(bounds.maxX-bounds.minX), (height-52)/(bounds.maxY-bounds.minY));
    const centerX=(bounds.minX+bounds.maxX)/2, centerY=(bounds.minY+bounds.maxY)/2;
    const px = (x) => width/2+(x-centerX)*scale, py = (y) => height/2-(y-centerY)*scale;
    sceneProjection={scale,px,py};
    function line(x1,y1,x2,y2,stroke,lineWidth=1) {
      ctx.beginPath(); ctx.moveTo(px(x1),py(y1)); ctx.lineTo(px(x2),py(y2));
      ctx.strokeStyle=stroke; ctx.lineWidth=lineWidth; ctx.stroke();
    }
    function dot(x,y,r,fill) {
      ctx.beginPath(); ctx.arc(px(x),py(y),r,0,Math.PI*2); ctx.fillStyle=fill; ctx.fill();
    }
    function rectangle(x,y,sx,sy,angle,fill,stroke) {
      ctx.save(); ctx.translate(px(x),py(y)); ctx.rotate(-angle);
      ctx.fillStyle=fill; ctx.strokeStyle=stroke; ctx.lineWidth=1.3;
      ctx.fillRect(-sx*scale/2,-sy*scale/2,sx*scale,sy*scale);
      ctx.strokeRect(-sx*scale/2,-sy*scale/2,sx*scale,sy*scale); ctx.restore();
    }
    function arrow(x,y,vx,vy,stroke) {
      const length=Math.hypot(vx,vy); if(length<.02) return;
      const ex=x+vx*.4, ey=y+vy*.4, angle=Math.atan2(vy,vx), head=.10;
      line(x,y,ex,ey,stroke,1.5);
      line(ex,ey,ex-head*Math.cos(angle-.45),ey-head*Math.sin(angle-.45),stroke,1.5);
      line(ex,ey,ex-head*Math.cos(angle+.45),ey-head*Math.sin(angle+.45),stroke,1.5);
    }
    ctx.font="11px system-ui"; ctx.fillStyle="#8a9bb0"; ctx.textAlign="center";
    const xStep=bounds.maxX-bounds.minX>16?2:1,yStep=bounds.maxY-bounds.minY>16?2:1;
    for(let x=Math.ceil(bounds.minX/xStep)*xStep;x<=bounds.maxX;x+=xStep) {line(x,bounds.minY,x,bounds.maxY,"#223044"); ctx.fillText(String(x),px(x),py(bounds.minY)+18);}
    ctx.textAlign="right";
    for(let y=Math.ceil(bounds.minY/yStep)*yStep;y<=bounds.maxY;y+=yStep) {line(bounds.minX,y,bounds.maxX,y,"#223044"); ctx.fillText(String(y),px(bounds.minX)-9,py(y)+4);}
    ctx.fillStyle="#9caec3"; ctx.fillText("x / m",px(bounds.maxX),py(bounds.minY)+33);
    ctx.fillText("y / m",px(bounds.minX)-6,py(bounds.maxY)-12);
    if(!data) {ctx.textAlign="center";ctx.fillText("Loading recording…",width/2,height/2);return;}
    const f=data.fields, n=manifest.obstacle_specs.length;
    const rx=f.robot_xy[frame*2], ry=f.robot_xy[frame*2+1], yaw=f.robot_yaw[frame];
    ctx.save(); ctx.beginPath(); ctx.rect(px(bounds.minX),py(bounds.maxY),(bounds.maxX-bounds.minX)*scale,(bounds.maxY-bounds.minY)*scale); ctx.clip();
    voView.drawBackground(ctx,sceneProjection);
    data.walls.forEach((wall) => rectangle(wall.x,wall.y,...wall.size,wall.yaw,"#344256","#64758c"));
    if($("showTrail").checked) {
      let first=frame; while(first>0 && f.episode_id[first-1]===f.episode_id[frame]) first--;
      ctx.beginPath();ctx.moveTo(px(f.robot_xy[first*2]),py(f.robot_xy[first*2+1]));
      for(let i=first+1;i<=frame;i++) ctx.lineTo(px(f.robot_xy[i*2]),py(f.robot_xy[i*2+1]));
      ctx.strokeStyle="#6dc9ec99";ctx.lineWidth=2;ctx.stroke();
    }
    for(let i=0;i<n;i++) {
      if(!f.obstacle_active[frame*n+i]) continue;
      const index=(frame*n+i)*2, x=f.obstacle_xy[index], y=f.obstacle_xy[index+1], spec=manifest.obstacle_specs[i];
      const tint=spec.name.startsWith("moving_") ? color.moving : color.static;
      if(spec.radius>0) {
        ctx.beginPath();ctx.arc(px(x),py(y),spec.radius*scale,0,Math.PI*2);
        ctx.fillStyle=tint+"28";ctx.fill();ctx.strokeStyle=tint;ctx.lineWidth=1.4;ctx.stroke();
      } else rectangle(x,y,2*spec.half_size[0],2*spec.half_size[1],0,tint+"28",tint);
      if($("showVelocity").checked) arrow(x,y,f.obstacle_velocity[index],f.obstacle_velocity[index+1],tint);
    }
    const lx=f.lidar_origin[2*frame], ly=f.lidar_origin[2*frame+1];
    for(let ray=0;ray<40;ray++) {
      const index=(frame*40+ray)*2, x=f.lidar_xy[index], y=f.lidar_xy[index+1], valid=f.lidar_valid[frame*40+ray];
      if($("showRays").checked) {
        if(!valid)ctx.setLineDash([3,4]);line(lx,ly,x,y,valid?"#75d7a74d":"#7896b026");ctx.setLineDash([]);
      }
      if($("showLidar").checked && valid) dot(x,y,2.5,color.lidar);
    }
    const gx=f.goal_xy[2*frame], gy=f.goal_xy[2*frame+1];
    ctx.save();ctx.translate(px(gx),py(gy));ctx.strokeStyle=color.goal;ctx.lineWidth=2;
    ctx.beginPath();ctx.moveTo(-7,0);ctx.lineTo(7,0);ctx.moveTo(0,-7);ctx.lineTo(0,7);ctx.stroke();ctx.restore();
    ctx.beginPath();ctx.arc(px(rx),py(ry),manifest.robot_radius*scale,0,Math.PI*2);
    ctx.strokeStyle="#6dc9ec55";ctx.lineWidth=1;ctx.setLineDash([3,3]);ctx.stroke();ctx.setLineDash([]);
    rectangle(rx,ry,.50,.27,yaw,"#6dc9ec44",color.robot);
    line(rx,ry,rx+.44*Math.cos(yaw),ry+.44*Math.sin(yaw),color.robot,2.5);
    dot(rx+.44*Math.cos(yaw),ry+.44*Math.sin(yaw),3,color.robot);
    if($("showVelocity").checked) arrow(rx,ry,f.robot_velocity[frame*2],f.robot_velocity[frame*2+1],color.robot);
    voView.drawForeground(ctx,sceneProjection);
    ctx.restore();
  }

  function fillEnvironments() {
    const old=Number($("envSelect").value), level=$("levelFilter").value;
    const entries=manifest.environments.filter((entry) => level==="all" || entry.level===Number(level));
    $("envSelect").replaceChildren(...entries.map((entry) => new Option(`${entry.id} · level ${entry.level}`,String(entry.id))));
    const next=entries.find((entry) => entry.id===old) ?? entries[0];
    $("envSelect").value=String(next.id); chooseEnvironment(next.id);
  }

  $("playButton").addEventListener("click", () => {
    if(!data)return;
    if(playing){pause();return;}
    if(frame>=end)setFrame(start);
    playing=true;lastTick=null;clock=frame;$("playButton").textContent="Pause";
  });
  $("prevButton").addEventListener("click",()=>{pause();setFrame(frame-1);});
  $("nextButton").addEventListener("click",()=>{pause();setFrame(frame+1);});
  $("startButton").addEventListener("click",()=>{pause();setFrame(start);});
  $("timeline").addEventListener("input",()=>{pause();setFrame(Number($("timeline").value));});
  $("envSelect").addEventListener("change",()=>chooseEnvironment(Number($("envSelect").value)));
  $("levelFilter").addEventListener("change",fillEnvironments);
  ["showLidar","showRays","showTrail","showVelocity"].forEach((id)=>$(id).addEventListener("change",draw));
  window.addEventListener("resize",draw);
  document.addEventListener("visibilitychange",()=>{lastTick=null;});
  document.addEventListener("keydown",(event)=>{
    if(["INPUT","SELECT","TEXTAREA","BUTTON","SUMMARY"].includes(event.target.tagName))return;
    if(event.code==="Space"){event.preventDefault();$("playButton").click();}
    if(event.code==="ArrowLeft"){event.preventDefault();$("prevButton").click();}
    if(event.code==="ArrowRight"){event.preventDefault();$("nextButton").click();}
  });
  function tick(now) {
    if(playing && data) {
      if(lastTick!==null) clock+=(now-lastTick)/1000*Number($("rateSelect").value)/manifest.dt;
      lastTick=now;
      if(clock>=end+1) {
        if($("playbackMode").value==="loop")clock=start+(clock-start)%(end-start+1);
        else {
          const select=$("envSelect");
          const next=(select.selectedIndex+1)%select.options.length;
          select.selectedIndex=next;
          chooseEnvironment(Number(select.value),true);
          requestAnimationFrame(tick);
          return;
        }
      }
      const next=Math.floor(clock);
      if(next!==frame){frame=Math.max(start,Math.min(end,next));$("timeline").value=frame;render();}
    } else lastTick=null;
    requestAnimationFrame(tick);
  }
  if(!manifest || !manifest.environments?.length) {error("Replay data not found. Open the exported index.html and keep the export folder intact.");return;}
  [...new Set(manifest.environments.map((entry)=>entry.level))].sort((a,b)=>a-b).forEach((level)=>$("levelFilter").add(new Option(String(level),String(level))));
  fillEnvironments();requestAnimationFrame(tick);
})();
