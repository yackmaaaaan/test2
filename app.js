const $=s=>document.querySelector(s); const log=(...a)=>{$('#log').textContent+=a.join(' ')+'\n'};
ort.env.wasm.wasmPaths='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/'; ort.env.wasm.numThreads=1;
let det,cls,classes,imgBitmap;
const JP={'0m':'赤5萬','0p':'赤5筒','0s':'赤5索','0z':'赤白','1z':'東','2z':'南','3z':'西','4z':'北','5z':'白','6z':'發','7z':'中','back':'裏'};
function label(c){if(JP[c])return JP[c]; const n=c[0],s=c[1]; return n+({m:'萬',p:'筒',s:'索'}[s]||s)}
async function loadModels(){try{$('#status').textContent='モデル読込中…'; classes=await (await fetch('models/mahjong-classes.json')).json(); const parts=await (await fetch('models/data-parts.json')).json(); let blobs=[];let total=0; for(let i=0;i<parts.length;i++){ $('#status').textContent=`分類器データ読込 ${i+1}/${parts.length}`; const b=await (await fetch('models/'+parts[i],{cache:'no-store'})).blob(); blobs.push(b); total+=b.size; } const ext=new Blob(blobs,{type:'application/octet-stream'}); blobs=[]; log('classifier external data', (total/1048576).toFixed(1),'MB'); det=await ort.InferenceSession.create('models/tile-yolo-26n.onnx',{executionProviders:['wasm']}); cls=await ort.InferenceSession.create('models/classifier-resnet50.onnx',{executionProviders:['wasm'],externalData:[{path:'classifier-resnet50.onnx.data',data:ext}]}); log('det input',JSON.stringify(det.inputNames),'output',JSON.stringify(det.outputNames)); log('cls input',JSON.stringify(cls.inputNames),'output',JSON.stringify(cls.outputNames)); $('#status').textContent='モデル読込完了。写真を選択してください。';}catch(e){$('#status').textContent='モデル読込失敗: '+e.message;log(e.stack||e)}}
$('#file').onchange=async e=>{const f=e.target.files[0];if(!f)return; imgBitmap=await createImageBitmap(f); const u=URL.createObjectURL(f); $('#preview').src=u;$('#preview').style.display='block';$('#run').disabled=!det||!cls;$('#status').textContent=`画像 ${imgBitmap.width}×${imgBitmap.height}`};
function letterbox(bitmap,size=640){const c=document.createElement('canvas');c.width=c.height=size;const x=c.getContext('2d');x.fillStyle='rgb(114,114,114)';x.fillRect(0,0,size,size);const sc=Math.min(size/bitmap.width,size/bitmap.height),w=bitmap.width*sc,h=bitmap.height*sc,dx=(size-w)/2,dy=(size-h)/2;x.drawImage(bitmap,dx,dy,w,h);const d=x.getImageData(0,0,size,size).data,arr=new Float32Array(3*size*size);for(let i=0;i<size*size;i++){arr[i]=d[i*4]/255;arr[size*size+i]=d[i*4+1]/255;arr[2*size*size+i]=d[i*4+2]/255}return{tensor:new ort.Tensor('float32',arr,[1,3,size,size]),sc,dx,dy}}
function sigmoid(x){return 1/(1+Math.exp(-x))}
function parseDet(t,meta){
  const d=t.data, sh=t.dims;
  log('det output shape',JSON.stringify(sh));
  if(sh.length!==3) throw new Error('未対応の検出出力形状: '+JSON.stringify(sh));

  // HaseLab tile-yolo-26n.onnx:
  // output0 = [1,37,8400]
  // 37 = cx,cy,w,h + 1 class logit + 32 mask coefficients.
  // IMPORTANT: channel 4 is a raw logit in this export, so sigmoid must ALWAYS be applied.
  // Treating 0 logits as an already-normalized probability caused thousands of false candidates at 0.500.
  // output1 = [1,32,160,160] is the segmentation prototype and is not
  // required when all we need is a rectangular crop of each tile.
  let N,C,at;
  if(sh[1]===37){ C=37; N=sh[2]; at=(n,c)=>d[c*N+n]; }
  else if(sh[2]===37){ N=sh[1]; C=37; at=(n,c)=>d[n*C+c]; }
  else throw new Error('HaseLab検出モデルのoutput0ではありません: '+JSON.stringify(sh));

  // Log the actual score distribution. This makes threshold tuning possible.
  let scores=[];
  for(let n=0;n<N;n++){
    let q=sigmoid(at(n,4)); scores.push(q);
  }
  scores.sort((a,b)=>b-a);
  log('top confidences', scores.slice(0,20).map(v=>v.toFixed(3)).join(', '));

  // A raw logit of 0 becomes 0.5 after sigmoid, so the threshold must be > 0.5.
  // 0.52 keeps weaker true tiles while rejecting the mass of zero-logit background candidates.
  const threshold=0.52;
  let rows=[];
  for(let n=0;n<N;n++){
    let x=at(n,0), y=at(n,1), w=at(n,2), h=at(n,3), score=sigmoid(at(n,4));
    if(score<threshold) continue;

    // Exported Ultralytics boxes are in 640x640 letterboxed pixel coordinates.
    let x1=(x-w/2-meta.dx)/meta.sc, y1=(y-h/2-meta.dy)/meta.sc;
    let x2=(x+w/2-meta.dx)/meta.sc, y2=(y+h/2-meta.dy)/meta.sc;
    x1=Math.max(0,x1); y1=Math.max(0,y1);
    x2=Math.min(imgBitmap.width,x2); y2=Math.min(imgBitmap.height,y2);
    const bw=x2-x1, bh=y2-y1;
    if(bw<8 || bh<8) continue;
    // Mahjong tiles should be reasonably compact; reject obvious scene-sized boxes.
    if(bw>imgBitmap.width*.45 || bh>imgBitmap.height*.65) continue;
    rows.push({x1,y1,x2,y2,score});
  }
  log('candidates >= '+threshold, rows.length);

  rows.sort((a,b)=>b.score-a.score);
  const iou=(a,b)=>{
    const x=Math.max(a.x1,b.x1), y=Math.max(a.y1,b.y1), X=Math.min(a.x2,b.x2), Y=Math.min(a.y2,b.y2);
    const inter=Math.max(0,X-x)*Math.max(0,Y-y), aa=(a.x2-a.x1)*(a.y2-a.y1), bb=(b.x2-b.x1)*(b.y2-b.y1);
    return inter/(aa+bb-inter+1e-9);
  };
  let keep=[];
  for(const r of rows){ if(!keep.some(k=>iou(k,r)>.50)) keep.push(r); }

  // A hand can contain 14 tiles (more with kans); allow some extra detections for diagnosis.
  keep=keep.slice(0,24);
  // Reading order: primarily left-to-right. For slightly staggered photos this is enough.
  keep.sort((a,b)=>((a.x1+a.x2)-(b.x1+b.x2)));
  log('after duplicate removal', keep.length);
  return keep;
}
function cropCanvas(r){const pad=.06*Math.max(r.x2-r.x1,r.y2-r.y1),sx=Math.max(0,r.x1-pad),sy=Math.max(0,r.y1-pad),sw=Math.min(imgBitmap.width-sx,r.x2-r.x1+2*pad),sh=Math.min(imgBitmap.height-sy,r.y2-r.y1+2*pad);const c=document.createElement('canvas');c.width=c.height=224;const x=c.getContext('2d');x.fillStyle='white';x.fillRect(0,0,224,224);const sc=Math.min(224/sw,224/sh),dw=sw*sc,dh=sh*sc;x.drawImage(imgBitmap,sx,sy,sw,sh,(224-dw)/2,(224-dh)/2,dw,dh);return c}
function classTensor(c){const d=c.getContext('2d').getImageData(0,0,224,224).data,a=new Float32Array(3*224*224),mean=[.485,.456,.406],std=[.229,.224,.225];for(let i=0;i<224*224;i++)for(let k=0;k<3;k++)a[k*224*224+i]=(d[i*4+k]/255-mean[k])/std[k];return new ort.Tensor('float32',a,[1,3,224,224])}
async function classify(c){const input=cls.inputNames[0],o=await cls.run({[input]:classTensor(c)}),v=o[cls.outputNames[0]].data;let mx=-Infinity,ix=0;for(let i=0;i<v.length;i++)if(v[i]>mx){mx=v[i];ix=i}let sum=0;for(const z of v)sum+=Math.exp(z-mx);return{idx:ix,conf:1/sum,code:classes[ix]}}
$('#run').onclick=async()=>{if(!imgBitmap)return;try{$('#run').disabled=true;$('#tiles').innerHTML='';$('#status').textContent='牌を検出中…';const meta=letterbox(imgBitmap);const out=await det.run({[det.inputNames[0]]:meta.tensor});const detTensor=out['output0'] || out[det.outputNames[0]]; log('using detector output', out['output0'] ? 'output0' : det.outputNames[0]); let boxes=parseDet(detTensor,meta);$('#count').textContent=`${boxes.length}枚`;log('detected',boxes.length);if(!boxes.length)throw new Error('牌を検出できませんでした。検出出力形状をログで確認してください。');for(let i=0;i<boxes.length;i++){ $('#status').textContent=`牌種を分類中 ${i+1}/${boxes.length}`;const c=cropCanvas(boxes[i]),p=await classify(c),card=document.createElement('div');card.className='tile';card.appendChild(c);card.insertAdjacentHTML('beforeend',`<b>${label(p.code)}</b><small>${p.code} / ${(p.conf*100).toFixed(1)}%<br>検出 ${(boxes[i].score*100).toFixed(1)}%</small>`);$('#tiles').appendChild(card)}$('#status').textContent=`完了：${boxes.length}枚検出`;}catch(e){$('#status').textContent='認識失敗: '+e.message;log(e.stack||e)}finally{$('#run').disabled=false}};
loadModels();