"""Self-contained, network-free WebGL panorama viewer."""

import base64
import json
from pathlib import Path


def viewer_html(image_path, canvas):
    photo = base64.b64encode(Path(image_path).read_bytes()).decode()
    return TEMPLATE.replace("__PHOTO__", photo).replace("__SETTINGS__", json.dumps(canvas))


TEMPLATE = r"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<style>
*{box-sizing:border-box}body{margin:0;background:#101923;color:#e9eff5;font:13px system-ui}
#wrap{position:relative;width:100%;height:420px;border-radius:0;overflow:hidden}
canvas{width:100%;height:100%;display:block;cursor:grab;touch-action:none}canvas:active{cursor:grabbing}
button{position:absolute;right:8px;top:8px;background:#5c307ddd;border:1px solid #ffffff44;border-radius:0;color:white;padding:5px 9px;cursor:pointer}
#wrap:fullscreen{height:100vh;border-radius:0}.error{padding:30px}
</style><div id="wrap"><canvas id="view" aria-label="拖动浏览全景，滚轮缩放"></canvas><button id="reset" style="right:64px">复位</button><button id="full">全屏</button></div>
<script>
const settings=__SETTINGS__,canvas=document.getElementById("view"),gl=canvas.getContext("webgl");
if(!gl){document.getElementById("wrap").innerHTML='<div class="error">此浏览器不支持 WebGL，请使用平面全景预览。</div>';}
else{
const vs=`attribute vec2 p;void main(){gl_Position=vec4(p,0.,1.);}`;
const fs=`precision highp float;uniform sampler2D image;uniform vec2 resolution;uniform vec4 bounds;uniform vec3 camera;
void main(){vec2 q=(gl_FragCoord.xy/resolution-.5)*2.;float t=tan(camera.z*.5);vec3 r=normalize(vec3(q.x*resolution.x/resolution.y*t,-q.y*t,1.));
float cp=cos(camera.y),sp=sin(camera.y);r=vec3(r.x,cp*r.y+sp*r.z,-sp*r.y+cp*r.z);
float cy=cos(camera.x),sy=sin(camera.x);r=vec3(cy*r.x+sy*r.z,r.y,-sy*r.x+cy*r.z);
float a=mod(atan(r.x,r.z)-bounds.x+6.2831853,6.2831853);float u=a/bounds.y;float v=(atan(r.y,length(r.xz))-bounds.z)/bounds.w;
if(u<0.||u>1.||v<0.||v>1.)gl_FragColor=vec4(.055,.085,.12,1.);else gl_FragColor=texture2D(image,vec2(u,v));}`;
function shader(type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s}
const program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,vs));gl.attachShader(program,shader(gl.FRAGMENT_SHADER,fs));gl.linkProgram(program);gl.useProgram(program);
const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,1,1]),gl.STATIC_DRAW);
const p=gl.getAttribLocation(program,"p");gl.enableVertexAttribArray(p);gl.vertexAttribPointer(p,2,gl.FLOAT,false,0,0);
const tex=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,tex);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);
let yaw=settings.theta_start+(settings.width/settings.scale)/2,pitch=0,fov=.85,ready=false,last=null;
const initialYaw=yaw;
function draw(){if(!ready)return;const dpr=Math.min(devicePixelRatio||1,2);canvas.width=canvas.clientWidth*dpr;canvas.height=canvas.clientHeight*dpr;gl.viewport(0,0,canvas.width,canvas.height);gl.uniform2f(gl.getUniformLocation(program,"resolution"),canvas.width,canvas.height);gl.uniform4f(gl.getUniformLocation(program,"bounds"),settings.theta_start,settings.width/settings.scale,settings.phi_start,settings.height/settings.scale);gl.uniform3f(gl.getUniformLocation(program,"camera"),yaw,pitch,fov);gl.drawArrays(gl.TRIANGLE_STRIP,0,4)}
const im=new Image();im.onload=()=>{const max=gl.getParameter(gl.MAX_TEXTURE_SIZE);let source=im;if(im.width>max||im.height>max){source=document.createElement("canvas");const k=Math.min(max/im.width,max/im.height);source.width=im.width*k;source.height=im.height*k;source.getContext("2d").drawImage(im,0,0,source.width,source.height)}gl.texImage2D(gl.TEXTURE_2D,0,gl.RGB,gl.RGB,gl.UNSIGNED_BYTE,source);ready=true;draw()};im.src="data:image/jpeg;base64,__PHOTO__";
canvas.onpointerdown=e=>{last=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};
canvas.onpointermove=e=>{if(!last)return;yaw-=(e.clientX-last[0])*.003;pitch=Math.max(-.55,Math.min(.55,pitch-(e.clientY-last[1])*.002));last=[e.clientX,e.clientY];draw()};
canvas.onpointerup=()=>last=null;canvas.onpointercancel=()=>last=null;
canvas.addEventListener("wheel",e=>{e.preventDefault();fov=Math.max(.3,Math.min(1.3,fov+e.deltaY*.0005));draw()},{passive:false});
function resetView(){yaw=initialYaw;pitch=0;fov=.85;draw()}
canvas.ondblclick=resetView;document.getElementById("reset").onclick=resetView;
document.getElementById("full").onclick=()=>{const wrap=document.getElementById("wrap");if(document.fullscreenElement)document.exitFullscreen();else wrap.requestFullscreen()};
new ResizeObserver(draw).observe(canvas);
}
</script></html>"""
