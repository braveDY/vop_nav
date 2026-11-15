(function(root) {
  "use strict";
  function intervals(values,frame) {
    const minimum=new Float32Array(360),maximum=new Float32Array(360),valid=new Uint8Array(360);
    for(let i=0;i<360;i++) {
      const index=(frame*360+i)*2;
      minimum[i]=values[index];maximum[i]=values[index+1];valid[i]=maximum[i]-minimum[i]>1e-8;
    }
    return {minimum,maximum,valid};
  }
  function metrics(prediction,reference) {
    let iou=0,mae=0,extra=0,missed=0;
    for(let i=0;i<360;i++) {
      const a=prediction.minimum[i],b=prediction.maximum[i],c=reference.minimum[i],d=reference.maximum[i];
      const intersection=Math.max(0,Math.min(b,d)-Math.max(a,c));
      const predicted=Math.max(0,b-a),truth=Math.max(0,d-c),union=predicted+truth-intersection;
      iou+=union<=1e-8?1:intersection/Math.max(union,1e-8);
      mae+=(Math.abs(a-c)+Math.abs(b-d))/2;extra+=predicted-intersection;missed+=truth-intersection;
    }
    return {iou:iou/360,mae:mae/360,extra:extra/360,missed:missed/360};
  }
  function subtract(a,b,c,d) {
    const pieces=[];
    if(Math.min(b,c)>a)pieces.push([a,Math.min(b,c)]);
    if(b>Math.max(a,d))pieces.push([Math.max(a,d),b]);
    return pieces;
  }
  const api={intervals,metrics,subtract};
  if(typeof module!=="undefined"&&module.exports)module.exports=api;
  else root.VOPComparison=api;
})(typeof window!=="undefined"?window:globalThis);
