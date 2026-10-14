const imageInput=document.getElementById('imageInput');
const preview=document.getElementById('preview');
const dropzone=document.getElementById('dropzone');
const analyzeButton=document.getElementById('analyzeButton');
const apiUrl=document.getElementById('apiUrl');
const statusText=document.getElementById('status');
let selectedFile=null;
document.getElementById('year').textContent=new Date().getFullYear();
function setStatus(message,isError=false){statusText.textContent=message;statusText.style.color=isError?'#ffaaa5':'var(--teal)'}
function chooseFile(file){
  if(!file)return;
  if(!['image/jpeg','image/png','image/webp'].includes(file.type)){setStatus('Please choose a JPG, PNG, or WEBP image.',true);return}
  if(file.size>10*1024*1024){setStatus('Image is larger than 10 MB. Please choose a smaller file.',true);return}
  selectedFile=file;preview.src=URL.createObjectURL(file);preview.hidden=false;
  document.getElementById('uploadTitle').textContent=file.name;
  document.getElementById('uploadHint').textContent='Image ready for analysis';
  document.getElementById('fileMeta').textContent=(file.size/1024/1024).toFixed(2)+' MB · '+file.type;
  analyzeButton.disabled=false;setStatus('');
}
imageInput.addEventListener('change',e=>chooseFile(e.target.files[0]));
['dragenter','dragover'].forEach(type=>dropzone.addEventListener(type,e=>{e.preventDefault();dropzone.classList.add('dragover')}));
['dragleave','drop'].forEach(type=>dropzone.addEventListener(type,e=>{e.preventDefault();dropzone.classList.remove('dragover')}));
dropzone.addEventListener('drop',e=>chooseFile(e.dataTransfer.files[0]));
function findPrediction(data){
  if(!data||typeof data!=='object')return {label:null,confidence:null};
  const labelKeys=['predicted_class','prediction','class_name','label','class','predicted_label'];
  const confidenceKeys=['confidence','confidence_score','probability','score'];
  let label=null,confidence=null;
  for(const key of labelKeys){if(typeof data[key]==='string'||typeof data[key]==='number'){label=String(data[key]);break}}
  for(const key of confidenceKeys){if(typeof data[key]==='number'){confidence=data[key];break}}
  if(!label&&Array.isArray(data.predictions)&&data.predictions.length){
    const first=data.predictions[0];
    if(typeof first==='string')label=first;
    else if(first&&typeof first==='object'){label=first.class_name||first.label||first.class||null;confidence=first.confidence??first.score??confidence}
  }
  return {label,confidence};
}
analyzeButton.addEventListener('click',async()=>{
  if(!selectedFile)return;
  const endpoint=apiUrl.value.trim();
  if(!endpoint){setStatus('Enter your deployed prediction endpoint first.',true);apiUrl.focus();return}
  let parsed;
  try{parsed=new URL(endpoint)}catch{setStatus('Please enter a valid full URL, including https://.',true);return}
  if(!['http:','https:'].includes(parsed.protocol)){setStatus('Endpoint must use HTTP or HTTPS.',true);return}
  analyzeButton.disabled=true;analyzeButton.textContent='Analyzing…';
  document.getElementById('apiStatus').textContent='Requesting…';
  try{
    const body=new FormData();body.append('file',selectedFile);
    const response=await fetch(endpoint,{method:'POST',body});
    const raw=await response.text();let data;
    try{data=JSON.parse(raw)}catch{data={response:raw}};
    document.getElementById('rawResponse').textContent=JSON.stringify(data,null,2);
    if(!response.ok)throw new Error('API returned HTTP '+response.status+'. Check the endpoint and request field name.');
    const result=findPrediction(data);
    document.getElementById('resultTitle').textContent=result.label||'Response received';
    document.getElementById('resultDescription').textContent=result.label?'Prediction returned by your model API. Verify the label against your model’s class mapping.':'The API responded successfully, but its JSON fields were not recognized automatically. See the raw response below.';
    document.getElementById('prediction').textContent=result.label||'See raw response';
    document.getElementById('confidence').textContent=result.confidence===null?'Not provided':(result.confidence<=1?(result.confidence*100).toFixed(2)+'%':result.confidence.toFixed(2));
    document.getElementById('apiStatus').textContent='HTTP '+response.status;
    setStatus('API response received.');
  }catch(error){
    document.getElementById('apiStatus').textContent='Request failed';
    document.getElementById('resultTitle').textContent='Could not complete prediction';
    document.getElementById('resultDescription').textContent='The API request failed. Check the endpoint, server status, CORS settings, and expected image field.';
    document.getElementById('rawResponse').textContent=error.message;
    setStatus(error.message+' If this is a browser CORS error, enable CORS on the API for your frontend origin.',true);
  }finally{analyzeButton.disabled=!selectedFile;analyzeButton.innerHTML='Analyze image <span>→</span>'}
});