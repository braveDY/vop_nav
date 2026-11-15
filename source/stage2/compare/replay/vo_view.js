"use strict";
window.VOPView = {
  overlayPoint(x,y,yaw,speed,direction,displayScale) {
    return [x+displayScale*speed*Math.cos(yaw+direction),y+displayScale*speed*Math.sin(yaw+direction)];
  },
  create(onChange) {
    const $=(id)=>document.getElementById(id);
    let data=null,manifest=null,frame=0,cached=null;
    let boundsData=null,boundsScale=null,bounds=null;
    let prediction=null,reference=null,analytic=null,modelReady=false,comparisonMetrics=null;
    let frameData=null,frameIndex=-1,configured=false;
    const mode=()=>$("sourceSelect").value;
    function parameters() {
      const labels=manifest.vo_config;
      return {horizon:labels.horizon_s,maximum:labels.maximum_speed_mps,radius:labels.robot_radius_m};
    }
    for(const [id,event] of [["displayScale","change"],
                            ["sourceSelect","change"],["showDifference","change"]])
      $(id).addEventListener(event,onChange);

    function update(nextData,nextManifest,nextFrame) {
      data=nextData;manifest=nextManifest;frame=nextFrame;
      if(!configured&&manifest) {
        const model=manifest.model;
        for(const option of $("sourceSelect").options)if(option.value!=="vo")option.disabled=!model;
        if(model) {
          $("sourceSelect").value="compare";

        }
        configured=true;
      }
      const locked=mode()!=="vo"&&Boolean(manifest?.model);
      $("comparisonPanel").hidden=!locked;
      $("showDifference").disabled=mode()!=="compare";
      const options=parameters();
      $("horizonValue").textContent=options.horizon.toFixed(1)+" s";
      $("voRadius").textContent=options.radius.toFixed(2)+" m";
      $("voMaximum").textContent=String(options.maximum);
      if(!data) {
        cached=null;
        return;
      }
      if(data!==frameData||frame!==frameIndex) {
        modelReady=Boolean(manifest.model&&data.fields.model_valid?.[frame]);
        prediction=modelReady?window.VOPComparison.intervals(data.fields.model_intervals,frame):null;
        reference=window.VOPComparison.intervals(data.fields.vo_reference,frame);
        comparisonMetrics=prediction?window.VOPComparison.metrics(prediction,reference):null;
        const yaw=data.fields.robot_yaw[frame],vx=data.fields.robot_velocity[frame*2],vy=data.fields.robot_velocity[frame*2+1];
        const currentVelocity=[Math.cos(yaw)*vx+Math.sin(yaw)*vy,-Math.sin(yaw)*vx+Math.cos(yaw)*vy];
        const speed=Math.hypot(...currentVelocity),angle=Math.atan2(currentVelocity[1],currentVelocity[0]);
        const sector=((Math.round(angle*180/Math.PI)+180)%360+360)%360;
        analytic={...reference,currentVelocity,speedLimit:options.maximum,
          currentUnsafe:!reference.valid[sector]||speed<reference.minimum[sector]-1e-6||speed>reference.maximum[sector]+1e-6};
        frameData=data;frameIndex=frame;
      }
      cached=mode()==="model"?(prediction?{...analytic,...prediction}:null):analytic;
      const speed=Math.hypot(...analytic.currentVelocity);
      $("currentVelocityStatus").textContent=`Speed ${speed.toFixed(2)} m/s · ${analytic.currentUnsafe?"Outside VO interval":"Within VO interval"}`+(speed>analytic.speedLimit?" · Over limit":"");
      $("currentVelocityStatus").classList.toggle("unsafe",analytic.currentUnsafe||speed>analytic.speedLimit);
      if(locked) {
        $("frameIou").textContent=comparisonMetrics?(comparisonMetrics.iou*100).toFixed(1)+"%":"—";
        for(const [id,key] of [["frameMae","mae"],["frameExtra","extra"],["frameMissed","missed"]])
          $(id).textContent=comparisonMetrics?comparisonMetrics[key].toFixed(3)+" m/s":"—";
      }
    }

    function getBounds() {
      if(!data)return null;
      const displayScale=Number($("displayScale").value);
      if(boundsData!==data||boundsScale!==displayScale) {
        let minX=-5.5,maxX=5.5,minY=-3,maxY=3;
        const r=manifest.vo_config.maximum_speed_mps*displayScale+.4,xy=data.fields.robot_xy;
        // Fix the view for the entire selected recording, preventing camera pumping.
        for(let i=0;i<data.frames;i++) {
          minX=Math.min(minX,xy[2*i]-r);maxX=Math.max(maxX,xy[2*i]+r);
          minY=Math.min(minY,xy[2*i+1]-r);maxY=Math.max(maxY,xy[2*i+1]+r);
        }
        bounds={minX,maxX,minY,maxY};boundsData=data;boundsScale=displayScale;
      }
      return bounds;
    }

    function drawing(projection) {
      if(!data||!cached)return null;
      const rx=data.fields.robot_xy[frame*2],ry=data.fields.robot_xy[frame*2+1],yaw=data.fields.robot_yaw[frame];
      const factor=Number($("displayScale").value),unit=projection.scale*factor;
      const point=(speed,theta)=>{
        const [x,y]=window.VOPView.overlayPoint(rx,ry,yaw,speed,theta,factor);
        return [projection.px(x),projection.py(y)];
      };
      return {rx,ry,yaw,unit,cx:projection.px(rx),cy:projection.py(ry),point};
    }

    function drawBackground(ctx,projection) {
      const view=drawing(projection);if(!view)return;
      const {cx,cy,unit,yaw,point}=view;
      function segment(low,high,theta,fill) {
        if(high-low<1e-9)return;
        const a=theta-Math.PI/360,b=theta+Math.PI/360;
        ctx.beginPath();ctx.moveTo(...point(low,a));ctx.lineTo(...point(high,a));
        ctx.arc(cx,cy,high*unit,-(yaw+a),-(yaw+b),true);
        ctx.lineTo(...point(low,b));
        if(low>0)ctx.arc(cx,cy,low*unit,-(yaw+b),-(yaw+a),false);
        ctx.closePath();ctx.fillStyle=fill;ctx.fill();
      }
      ctx.save();
      for(let i=0;i<360;i++) {
        const theta=(i-180)*Math.PI/180;
        if(cached.valid[i])segment(cached.minimum[i],cached.maximum[i],theta,mode()==="model"?"#c397ef40":"#65cf9446");
        if(mode()==="compare"&&prediction&&$("showDifference").checked) {
          const a=prediction.minimum[i],b=prediction.maximum[i],c=reference.minimum[i],d=reference.maximum[i];
          for(const interval of window.VOPComparison.subtract(a,b,c,d))segment(...interval,theta,"#fa94946b");
          for(const interval of window.VOPComparison.subtract(c,d,a,b))segment(...interval,theta,"#79b7ff66");
        }
      }
      function boundary(intervals,values,color,dashed=false) {
        ctx.beginPath();let open=false;
        for(let i=0;i<=360;i++) {
          const index=i%360;
          if(!intervals.valid[index]){open=false;continue;}
          const p=point(values[index],(i-180)*Math.PI/180);
          if(open)ctx.lineTo(...p);else ctx.moveTo(...p);open=true;
        }
        ctx.strokeStyle=color;ctx.lineWidth=1.4;ctx.setLineDash(dashed?[4,3]:[]);ctx.stroke();ctx.setLineDash([]);
      }
      boundary(cached,cached.minimum,mode()==="model"?"#d5a6ff":"#edac65aa",mode()==="model");
      boundary(cached,cached.maximum,mode()==="model"?"#d5a6ff":"#75d7a7bb",mode()==="model");
      if(mode()==="compare"&&prediction)for(const values of [prediction.minimum,prediction.maximum])boundary(prediction,values,"#d5a6ff",true);
      ctx.beginPath();ctx.arc(cx,cy,cached.speedLimit*unit,0,2*Math.PI);
      ctx.strokeStyle="#82aaffcc";ctx.lineWidth=1.4;ctx.setLineDash([6,5]);ctx.stroke();ctx.setLineDash([]);
      ctx.font="11px system-ui";ctx.textAlign="center";ctx.fillStyle="#9cbbff";
      ctx.fillText(`${cached.speedLimit} m/s`,cx,cy-cached.speedLimit*unit-7);
      ctx.restore();
    }

    function drawForeground(ctx,projection) {
      const view=drawing(projection);if(!view)return;
      const {cx,cy,point}=view;
      ctx.save();
      const [vx,vy]=cached.currentVelocity,speed=Math.hypot(vx,vy),angle=Math.atan2(vy,vx);
      const end=point(Math.min(speed,cached.speedLimit),angle);
      ctx.lineWidth=1.7;ctx.strokeStyle=cached.currentUnsafe?"#ffb3aa":"#f4f7ff";
      ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(...end);ctx.stroke();
      ctx.beginPath();ctx.arc(...end,3,0,2*Math.PI);ctx.fillStyle=ctx.strokeStyle;ctx.fill();
      ctx.restore();
    }

    return {update,getBounds,drawBackground,drawForeground};
  }
};
