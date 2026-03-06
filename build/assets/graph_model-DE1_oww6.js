import{o as T,g as S,d as g,c as ae,E as v,dV as Vt,bm as ge,m as q,a_ as X,cM as U,ck as te,cI as pe,d7 as _e,ea as Pt,ec as Ft,ez as jt,h4 as xt,f4 as Rt,a9 as O,a2 as ce,fZ as Bt,fm as Ht,az as Z,cn as ee,h5 as We,fs as Wt,cp as re,fH as qt,fI as Ut,fJ as Gt,h6 as Ne,h7 as Kt,bh as nt,cv as it,cw as ot,cC as ne,aA as qe,h8 as be,b as Te,h9 as we,ha as ut,fY as Jt,w as Xt,hb as pt,d0 as mt,bS as lt,hc as Zt,cF as B,d4 as me,bE as ct,cr as dt,fX as Yt,gf as Qt,eU as Mt,hd as ea,a3 as Y,he as ta,bC as ht,hf as aa,hg as sa,hh as ra,hi as na,hj as ia,hk as oa,aG as Ue,gE as Ge,hl as ua,hm as pa,hn as ma,ho as la,hp as ca,hq as da,hr as ha,hs as yt,ht as Ke,hu as ya,a1 as Q,hv as ft,hw as fa,hx as gt,hy as ga,hz as Na,hA as ba,hB as Ta,hC as Nt,hD as wa,hE as Sa,hF as bt,hG as Oa,hH as Tt,hI as va,hJ as _a,hK as Aa,hL as Ea,hM as Ia,hN as ka,hO as $a,hP as Da,hQ as za,j as Fe,dJ as wt,eD as St,gA as Je,hR as La,bk as Ot,hS as L,di as Ca,aX as Va,aY as Pa,aZ as Fa,a$ as ja,b0 as xa,b1 as Ra,b2 as Ba,b3 as Ha,b4 as Wa,b5 as qa,b6 as Ua,b7 as Ga,b8 as Ka,b9 as Ja,bb as Xa,bc as Za,bd as Ya,be as Qa,ba as Ma,bf as es,bg as ts,bi as as,bj as ss,bl as rs,bn as ns,bo as is,bp as os,bq as us,br as ps,bs as ms,bt as ls,bu as cs,bv as ds,bw as hs,bx as ys,dm as fs,by as gs,bz as Ns,bA as bs,bB as Ts,bD as ws,bF as Ss,bG as Os,dk as vs,bH as _s,bI as As,dl as Es,bJ as Is,bK as ks,bL as $s,bM as Ds,a7 as zs,bN as Ls,bO as Cs,cU as Vs,bP as Ps,bQ as Fs,bR as js,bT as xs,bU as Rs,cV as Bs,bV as Hs,a8 as Ws,cW as qs,bW as Us,bX as Gs,bY as Ks,bZ as Js,b_ as Xs,b$ as Zs,aR as Ys,c0 as Qs,c1 as Ms,c2 as er,c3 as tr,c4 as ar,c5 as sr,c6 as rr,c7 as nr,c8 as ir,c9 as or,aS as ur,ca as pr,cb as mr,cc as lr,cd as cr,ce as dr,cf as hr,cg as yr,ch as fr,ci as gr,cj as Nr,cl as br,dj as Tr,cm as wr,h as Sr,co as Or,cq as vr,cs as _r,ct as Ar,cu as Er,cx as Ir,cy as kr,cz as $r,cA as Dr,cB as zr,cX as Lr,cD as Cr,cE as Vr,cG as Pr,cH as Fr,cJ as jr,aU as xr,cK as Rr,cL as Br,cN as Hr,cO as Wr,cP as qr,cQ as Ur,cR as Gr,cS as Kr,cT as Jr,aV as Xr,aT as Zr,cY as Yr,cZ as Qr,c_ as Mr,c$ as en,d1 as se,d2 as tn,d3 as an,aW as sn,d5 as rn,d6 as nn,a4 as on,a5 as un,d8 as pn,d9 as mn,t as ln,da as cn,db as dn,dc as hn,dd as ie,de as yn,df as fn,dg as gn,dh as Nn,dy as R,a6 as P,hT as de,dx as bn}from"./index-BmNsTOCG.js";/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Tn(a){S(Array.isArray(a),()=>"The argument passed to tf.addN() must be a list of tensors"),S(a.length>=1,()=>`Must pass at least one tensor to tf.addN(), but got ${a.length}`);const e=a.map((r,i)=>g(r,`tensors${i}`,"addN")),t=e[0];e.forEach(r=>{if(r.dtype!==t.dtype)throw new Error("All tensors passed to tf.addN() must have the same dtype")}),e.forEach(r=>{if(!ae(r.shape,t.shape))throw new Error("All tensors passed to tf.addN() must have the same shape")});const s=e;return v.runKernel(Vt,s)}const wn=T({addN_:Tn});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Sn(a,e,t,s,r,i){const u=g(a,"forgetBias","basicLSTMCell"),o=g(e,"lstmKernel","basicLSTMCell"),p=g(t,"lstmBias","basicLSTMCell"),m=g(s,"data","basicLSTMCell"),l=g(r,"c","basicLSTMCell"),c=g(i,"h","basicLSTMCell"),d=ge([m,c],1),h=q(d,o),N=X(h,p),f=N.shape[0],y=N.shape[1]/4,b=[f,y],_=U(N,[0,0],b),k=U(N,[0,y],b),w=U(N,[0,y*2],b),I=U(N,[0,y*3],b),D=X(te(pe(_),_e(k)),te(l,pe(X(u,w)))),z=te(_e(D),pe(I));return[D,z]}const On=T({basicLSTMCell_:Sn});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vn(a,e){const t=g(a,"x","bitwiseAnd"),s=g(e,"y","bitwiseAnd");if(!ae(t.shape,s.shape))throw new Error(`BitwiseAnd: Tensors must have the same shape. x: ${t.shape}, y: ${s.shape}`);if(t.dtype!=="int32"||s.dtype!=="int32")throw new Error(`BitwiseAnd: Only supports 'int32' values in tensor, found type of x: ${t.dtype} and type of y: ${s.dtype}`);const r={a:t,b:s};return v.runKernel(Pt,r)}const _n=T({bitwiseAnd_:vn});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function An(a,e){const t=g(a,"s0","broadcastArgs","int32"),s=g(e,"s1","broadcastArgs","int32");if(t.rank!==1)throw new Error(`broadcastArgs(): first input must be a vector (rank=1). Has rank ${t.rank}`);if(s.rank!==1)throw new Error(`broadcastArgs(): second input must be a vector (rank=1). Has rank ${s.rank}`);const r={s0:t,s1:s};return v.runKernel(Ft,r)}const En=T({broadcastArgs_:An});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function In(a){const t={x:g(a,"x","diag")};return v.runKernel(jt,t)}const kn=T({diag_:In});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $n(a,e){const t=g(a,"x","ensureShape","string_or_numeric");if(!xt(t.shape,e))throw new Error(`EnsureShape: Shape of tensor ${t.shape} is not compatible with expected shape ${e}`);return a}const Dn=T({ensureShape_:$n});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zn(a,e,t){if(t<=0)throw new Error("The number of values should be positive.");const s={start:a,stop:e,num:t};return v.runKernel(Rt,{},s)}/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const oe=2147483648;function Ln(a,e,t="left"){const s=g(a,"sortedSequence","searchSorted"),r=g(e,"values","searchSorted"),i=s.shape[s.shape.length-1],u=r.shape[r.shape.length-1],o=O(s,[-1,i]),p=O(r,[-1,u]);if(o.rank<2)throw new Error("Sorted input argument must be at least 2-dimensional");if(o.shape[0]!==p.shape[0])throw new Error("Leading dimension of 'sortedSequence' and 'values' must match.");if(ce(p.shape)>=oe)throw new Error(`values tensor size must less than ${oe}`);if(o.shape[1]>=oe)throw new Error(`trailing dim_size must less than ${oe} for int32 output type, was ${o.shape[1]}`);const m={sortedSequence:o,values:p},l={side:t};return v.runKernel(Bt,m,l)}const je=T({searchSorted_:Ln});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Cn(a,e){return je(a,e,"left")}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vn(a,e,t,s,r=!1){const u={x:g(a,"x","maxPoolWithArgmax")},o={filterSize:e,strides:t,pad:s,includeBatchInIndex:r},p=v.runKernel(Ht,u,o);return{result:p[0],indexes:p[1]}}const Pn=T({maxPoolWithArgmax_:Vn});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Fn(a,e,{indexing:t="xy"}={}){if(t!=="xy"&&t!=="ij")throw new TypeError(`${t} is not a valid third argument to meshgrid`);if(a===void 0)return[];let s=g(a,"x","meshgrid",a instanceof Z?a.dtype:"float32");if(e===void 0)return[s];let r=g(e,"y","meshgrid",e instanceof Z?e.dtype:"float32");const i=ce(s.shape),u=ce(r.shape);return t==="xy"?(s=O(s,[1,-1]),r=O(r,[-1,1]),[q(ee([u,1],s.dtype),s),q(r,ee([1,i],r.dtype))]):(s=O(s,[-1,1]),r=O(r,[1,-1]),[q(s,ee([1,u],s.dtype)),q(ee([i,1],r.dtype),r)])}function jn(a,e,t,s){const r=g(e,"data","multiRNNCell"),i=We(t,"c","multiRNNCell"),u=We(s,"h","multiRNNCell");let o=r;const p=[];for(let c=0;c<a.length;c++){const d=a[c](o,i[c],u[c]);p.push(d[0]),p.push(d[1]),o=d[1]}const m=[],l=[];for(let c=0;c<p.length;c+=2)m.push(p[c]),l.push(p[c+1]);return[m,l]}const xn=T({multiRNNCell_:jn});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Rn(a,e,t,s=!1){const r=g(a,"logits","multinomial"),i=r.size,u=r.rank;if(i<2)throw new Error(`Error in multinomial: you need at least 2 outcomes, but got ${i}.`);if(u>2)throw new Error(`Rank of probabilities must be 1 or 2, but is ${u}`);t=t||Math.random();const p={logits:u===1?O(r,[1,-1]):r},m={numSamples:e,seed:t,normalized:s},l=v.runKernel(Wt,p,m);return u===1?O(l,[l.size]):l}const Bn=T({multinomial_:Rn});function Hn(a,e){const t=g(a,"v1","outerProduct"),s=g(e,"v2","outerProduct");S(t.rank===1&&s.rank===1,()=>`Error in outerProduct: inputs must be rank 1, but got ranks ${t.rank} and ${s.rank}.`);const r=O(t,[-1,1]),i=O(s,[1,-1]);return q(r,i)}const Wn=T({outerProduct_:Hn});function qn(a,e,t=0){return S(e.length===2,()=>"Invalid number of paddings. Must be length of 2."),re(a,[e],t)}const Un=T({pad1d_:qn});function Gn(a,e,t=0){return S(e.length===2&&e[0].length===2&&e[1].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),re(a,e,t)}const Kn=T({pad2d_:Gn});function Jn(a,e,t=0){return S(e.length===3&&e[0].length===2&&e[1].length===2&&e[2].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),re(a,e,t)}const Xn=T({pad3d_:Jn});function Zn(a,e,t=0){return S(e.length===4&&e[0].length===2&&e[1].length===2&&e[2].length===2&&e[3].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),re(a,e,t)}const Yn=T({pad4d_:Zn});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qn(a,e,t,s){const r=a.map((l,c)=>g(l,`tensors${c}`,"raggedGather","int32")),i=g(e,"paramsDenseValues","raggedGather"),u=g(t,"indices","raggedGather","int32"),o={paramsNestedSplits:r,paramsDenseValues:i,indices:u},p={outputRaggedRank:s},m=v.runKernel(qt,o,p);return{outputNestedSplits:m.slice(0,m.length-1),outputDenseValues:m[m.length-1]}}const Mn=T({raggedGather_:Qn});/**
 * @license
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ei(a,e,t){const s=g(a,"starts","raggedRange"),r=g(e,"limits","raggedRange",s.dtype),i=g(t,"deltas","raggedRange",s.dtype),u={starts:s,limits:r,deltas:i},o=v.runKernel(Ut,u);return{rtNestedSplits:o[0],rtDenseValues:o[1]}}const ti=T({raggedRange_:ei});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ai(a,e,t,s,r){const i=g(a,"shape","raggedTensorToTensor","int32"),u=g(e,"values","raggedTensorToTensor"),o=g(t,"defaultValue","raggedTensorToTensor",u.dtype),p=s.map((c,d)=>g(c,`tensors${d}`,"raggedTensorToTensor","int32")),m={shape:i,values:u,defaultValue:o,rowPartitionTensors:p},l={rowPartitionTypes:r};return v.runKernel(Gt,m,l)}const si=T({raggedTensorToTensor_:ai});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ri(a,e,t){Ne(a);const s=ce(a);let r=null;if(t==null||t==="float32")r=new Float32Array(s);else if(t==="int32")r=new Int32Array(s);else if(t==="bool")r=new Uint8Array(s);else throw new Error(`Unknown data type ${t}`);for(let i=0;i<s;i++)r[i]=e();return v.makeTensor(r,a,t)}const ni=T({rand_:ri});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ii(a,e,t=1,s="float32",r){if(Ne(a),t==null&&(t=1),s==null&&(s="float32"),s!=="float32"&&s!=="int32")throw new Error(`Unsupported data type ${s}`);const i=new Kt(e,t,s,r),u=nt(a,s);for(let o=0;o<u.values.length;o++)u.values[o]=i.nextValue();return u.toTensor()}const oi=T({randomGamma_:ii});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ui(a,e,t){if(e!=null&&e==="bool")throw new Error(`Unsupported data type ${e}`);return it(a,0,1,e,t)}const pi=T({randomStandardNormal_:ui});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function mi(a,e,t,s){return ot(a,e,t,"int32",s)}const li=T({randomUniformInt_:mi});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ci(a){const e=g(a,"x","reverse");return S(e.rank===1,()=>`Error in reverse1D: x must be rank 1 but got rank ${e.rank}.`),ne(e,0)}const di=T({reverse1d_:ci});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hi(a,e){const t=g(a,"x","reverse");return S(t.rank===2,()=>`Error in reverse2D: x must be rank 2 but got rank ${t.rank}.`),ne(t,e)}const yi=T({reverse2d_:hi});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fi(a,e){const t=g(a,"x","reverse");return S(t.rank===3,()=>`Error in reverse3D: x must be rank 3 but got rank ${t.rank}.`),ne(t,e)}const gi=T({reverse3d_:fi});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ni(a,e){const t=g(a,"x","reverse");return S(t.rank===4,()=>`Error in reverse4D: x must be rank 4 but got rank ${t.rank}.`),ne(t,e)}const bi=T({reverse4d_:Ni});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function Ti(a,e){const t=g(a,"x","setdiff1d"),s=g(e,"y","setdiff1d");S(t.dtype===s.dtype,()=>`x and y should have the same dtype, but got x (${t.dtype}) and y (${s.dtype}).`),S(t.rank===1,()=>`x should be 1D tensor, but got x (${t.shape}).`),S(s.rank===1,()=>`y should be 1D tensor, but got y (${s.shape}).`);const r=await t.data(),i=await s.data(),u=new Set(i);let o=0;for(let l=0;l<r.length;l++)u.has(r[l])||o++;const p=new qe([o],t.dtype),m=new qe([o],"int32");for(let l=0,c=0;l<r.length;l++)u.has(r[l])||(p.values[c]=r[l],m.values[c]=l,c++);return[p.toTensor(),m.toTensor()]}const wi=Ti;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vt(a,e,t){if(be(a),e!=null&&e.length!==3)throw new Error("tensor3d() requires shape to have three numbers");const s=Te(a,t);if(s.length!==3&&s.length!==1)throw new Error("tensor3d() requires values to be number[][][] or flat/TypedArray");if(s.length===1&&e==null)throw new Error("tensor3d() requires shape to be provided when `values` are a flat array");return we(a,e,s,t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Si(a,e,t){if(be(a),e!=null&&e.length!==4)throw new Error("tensor4d() requires shape to have four numbers");const s=Te(a,t);if(s.length!==4&&s.length!==1)throw new Error("tensor4d() requires values to be number[][][][] or flat/TypedArray");if(s.length===1&&e==null)throw new Error("tensor4d() requires shape to be provided when `values` are a flat array");return we(a,e,s,t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Oi(a,e,t){if(be(a),e!=null&&e.length!==5)throw new Error("tensor5d() requires shape to have five numbers");const s=Te(a,t);if(s.length!==5&&s.length!==1)throw new Error("tensor5d() requires values to be number[][][][][] or flat/TypedArray");if(s.length===1&&e==null)throw new Error("tensor5d() requires shape to be provided when `values` are a flat array");return we(a,e,s,t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vi(a,e,t){if(be(a),e!=null&&e.length!==6)throw new Error("tensor6d() requires shape to have six numbers");const s=Te(a,t);if(s.length!==6&&s.length!==1)throw new Error("tensor6d() requires values to be number[][][][][][] or flat/TypedArray");if(s.length===1&&e==null)throw new Error("tensor6d() requires shape to be provided when `values` are a flat array");return e=e||s,we(a,e,s,t)}/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _i(a,e,t){const s=g(a,"tensor","tensorScatterupdate"),r=g(e,"indices","tensorScatterupdate","int32"),i=g(t,"updates","tensorScatterupdate");if(ut(i,r,s.shape),s.dtype!==i.dtype)throw new Error(`tensor and updates must have the same dtype, instead they are ${s.dtype} and ${i.dtype}.`);const u={tensor:s,indices:r,updates:i},o={};return v.runKernel(Jt,u,o)}const Ai=T({tensorScatterUpdate_:_i});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ei(a,e){return je(a,e,"right")}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function Ii(a){const e=g(a,"condition","whereAsync","bool"),t=await e.data(),s=Xt(e.shape,t);return a!==e&&e.dispose(),s}const _t=Ii;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function ki(a,e,t){const s=g(a,"tensor","boolMask"),r=g(e,"mask","boolMask","bool"),i=t??0,u=r.rank,o=s.shape;S(u>0,()=>"mask cannot be scalar"),pt(o.slice(i,i+u),r.shape,"mask's shape must match the first K dimensions of tensor's shape,");let p=1;for(let f=i;f<i+u;f++)p*=o[f];const m=o.slice(0,i).concat([p],o.slice(i+u)),l=O(s,m),c=O(r,[-1]),d=await _t(c),h=mt(d,[1]),N=lt(l,h,i);return a!==s&&s.dispose(),e!==r&&r.dispose(),h.dispose(),l.dispose(),c.dispose(),d.dispose(),N}const $i=ki;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Di(a,e,t,s,r=!0){const i=g(a,"v","movingAverage"),u=g(e,"x","movingAverage"),o=g(t,"decay","movingAverage");Zt(i,u),S(ae(i.shape,u.shape),()=>"Shape mismatch in v and x");const p=B(1),m=me(p,o);let l=te(me(u,i),m);if(r){S(s!=null,()=>"When using zeroDebias: true, step is required.");const c=g(s,"step","movingAverage");l=ct(l,me(p,dt(o,c)))}return X(i,l)}const zi=T({movingAverage_:Di});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Li(a,e,t){Ne(t);const s=g(a,"indices","scatterND","int32"),r=g(e,"updates","scatterND");ut(r,s,t);const i={indices:s,updates:r},u={shape:t};return v.runKernel(Yt,i,u)}const Ci=T({scatterND_:Li});function Vi(a,e,t,s){if(a.dtype!=="int32")throw new Error(`tf.sparseToDense() expects the indices to be int32 type, but the dtype was ${a.dtype}.`);if(a.rank>2)throw new Error(`sparseIndices should be a scalar, vector, or matrix, but got shape ${a.shape}.`);const r=a.rank>0?a.shape[0]:1,i=a.rank>1?a.shape[1]:1;if(t.length!==i)throw new Error(`outputShape has incorrect number of elements:, ${t.length}, should be: ${i}.`);const u=e.size;if(!(e.rank===0||e.rank===1&&u===r))throw new Error(`sparseValues has incorrect shape ${e.shape}, should be [] or [${r}]`);if(e.dtype!==s.dtype)throw new Error("sparseValues.dtype must match defaultValues.dtype")}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Pi(a,e,t,s=0){Ne(t);const r=g(a,"sparseIndices","sparseToDense","int32"),i=g(e,"sparseValues","sparseToDense","string_or_numeric"),u=g(s,"defaultValue","sparseToDense",i.dtype);Vi(r,i,t,u);const o={sparseIndices:r,sparseValues:i,defaultValue:u},p={outputShape:t};return v.runKernel(Qt,o,p)}const Fi=T({sparseToDense_:Pi});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ji(a,e){const t=g(e,"indices","gatherND","int32"),r={params:g(a,"x","gatherND","string_or_numeric"),indices:t};return v.runKernel(Mt,r)}const xi=T({gatherND_:ji});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function Ri(a,e,t=1){const s=g(a,"predictions","inTopK"),r=g(e,"targets","inTopK");S(s.rank>1,()=>`inTopK() expects the predictions to be of rank 2 or higher, but got ${s.rank}`),S(s.rank-1===r.rank,()=>`predictions rank should be 1 larger than targets rank, but got predictions rank ${s.rank} and targets rank ${r.rank}`),pt(s.shape.slice(0,s.shape.length-1),r.shape,"predictions's shape should be align with the targets' shape, except the last dimension.");const i=s.shape[s.shape.length-1];S(t>0&&t<=i,()=>`'k' passed to inTopK() must be > 0 && <= the predictions last dimension (${i}), but got ${t}`);const u=await s.data(),o=await r.data(),[p,m]=[u.length/i,i],l=ea("bool",p);for(let c=0;c<p;c++){const d=c*m,h=u.subarray(d,d+m),N=[];for(let f=0;f<h.length;f++)N.push({value:h[f],index:f});N.sort((f,y)=>y.value-f.value),l[c]=0;for(let f=0;f<t;f++)if(N[f].index===o[c]){l[c]=1;break}}return a!==s&&s.dispose(),e!==r&&r.dispose(),Y(l,r.shape,"bool")}const Bi=Ri;/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hi({x:a,filter:e,strides:t,pad:s,dataFormat:r="NHWC",dilations:i=[1,1],dimRoundingMode:u,bias:o,activation:p="linear",preluActivationWeights:m,leakyreluAlpha:l}){if(ta(v.state.gradientDepth,p)===!1){let I=ht(a,e,t,s,r,i,u);return o!=null&&(I=X(I,o)),aa(I,p,m,l)}const c=g(a,"x","depthwiseConv2d","float32"),d=g(e,"filter","depthwiseConv2d","float32");let h=c,N=!1;c.rank===3&&(N=!0,h=O(c,[1,c.shape[0],c.shape[1],c.shape[2]])),S(h.rank===4,()=>`Error in fused depthwiseConv2d: input must be rank 4, but got rank ${h.rank}.`),S(d.rank===4,()=>`Error in fused depthwiseConv2d: filter must be rank 4, but got rank ${d.rank}.`),S(h.shape[3]===d.shape[2],()=>`Error in fused depthwiseConv2d: number of input channels (${h.shape[3]}) must match the inChannels dimension in filter ${d.shape[2]}.`),i==null&&(i=[1,1]),S(sa(t,i),()=>`Error in fused depthwiseConv2d: Either strides or dilations must be 1. Got strides ${t} and dilations '${i}'`),ra("fused depthwiseConv2d",s,u);const f=na(h.shape,d.shape,t,i,s,u,!0);let y;o!=null&&(y=g(o,"bias","fused conv2d"),[y]=ia(y,c),oa(f.outShape,y.shape));let b;m!=null&&(b=g(m,"prelu weights","fused depthwiseConv2d"));const _=(I,D)=>{S(ua(i),()=>`Error in gradient of fused depthwiseConv2d: dilation rates greater than 1 are not yet supported. Got dilations '${i}'`);const[z,J,V,F]=D,Se=pa(I,V,p),Be=ma(J.shape,Se,z,t,s,i,u),He=la(J,Se,z.shape,t,s,i,u);if(F!=null){const Ct=ca(y,Se);return[Be,He,Ct]}return[Be,He]},k={x:h,filter:d,bias:y,preluActivationWeights:b},w={strides:t,pad:s,dataFormat:r,dilations:i,dimRoundingMode:u,activation:p,leakyreluAlpha:l};return o==null?Ue((D,z,J)=>{let V=v.runKernel(Ge,k,w);return J([z,D,V]),N&&(V=O(V,[V.shape[1],V.shape[2],V.shape[3]])),{value:V,gradFunc:_}})(h,d):Ue((D,z,J,V)=>{let F=v.runKernel(Ge,k,w);return V([z,D,F,J]),N&&(F=O(F,[F.shape[1],F.shape[2],F.shape[3]])),{value:F,gradFunc:_}})(h,d,y)}const Wi=T({fusedDepthwiseConv2d_:Hi});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const qi=Object.freeze(Object.defineProperty({__proto__:null,conv2d:da,depthwiseConv2d:Wi,matMul:ha},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ui="model",Gi=".json",Ki=".weights.bin";function Xe(a){return new Promise(e=>setTimeout(e)).then(a)}class G{constructor(e){if(!Q().getBool("IS_BROWSER"))throw new Error("browserDownloads() cannot proceed because the current environment is not a browser.");e.startsWith(G.URL_SCHEME)&&(e=e.slice(G.URL_SCHEME.length)),(e==null||e.length===0)&&(e=Ui),this.modelJsonFileName=e+Gi,this.weightDataFileName=e+Ki}async save(e){if(typeof document>"u")throw new Error("Browser downloads are not supported in this environment since `document` is not present");const t=ft.join(e.weightData),s=window.URL.createObjectURL(new Blob([t],{type:"application/octet-stream"}));if(e.modelTopology instanceof ArrayBuffer)throw new Error("BrowserDownloads.save() does not support saving model topology in binary formats yet.");{const r=[{paths:["./"+this.weightDataFileName],weights:e.weightSpecs}],i=fa(e,r),u=window.URL.createObjectURL(new Blob([JSON.stringify(i)],{type:"application/json"})),o=this.modelJsonAnchor==null?document.createElement("a"):this.modelJsonAnchor;if(o.download=this.modelJsonFileName,o.href=u,await Xe(()=>o.dispatchEvent(new MouseEvent("click"))),e.weightData!=null){const p=this.weightDataAnchor==null?document.createElement("a"):this.weightDataAnchor;p.download=this.weightDataFileName,p.href=s,await Xe(()=>p.dispatchEvent(new MouseEvent("click")))}return{modelArtifactsInfo:gt(e)}}}}G.URL_SCHEME="downloads://";class Ji{constructor(e){if(e==null||e.length<1)throw new Error(`When calling browserFiles, at least 1 file is required, but received ${e}`);this.jsonFile=e[0],this.weightsFiles=e.slice(1)}async load(){return new Promise((e,t)=>{const s=new FileReader;s.onload=r=>{const i=JSON.parse(r.target.result),u=i.modelTopology;if(u==null){t(new Error(`modelTopology field is missing from file ${this.jsonFile.name}`));return}if(i.weightsManifest==null){t(new Error(`weightManifest field is missing from file ${this.jsonFile.name}`));return}if(this.weightsFiles.length===0){e({modelTopology:u});return}const p=yt(i,m=>this.loadWeights(m));e(p)},s.onerror=r=>t(`Failed to read model topology and weights manifest JSON from file '${this.jsonFile.name}'. BrowserFiles supports loading Keras-style tf.Model artifacts only.`),s.readAsText(this.jsonFile)})}loadWeights(e){const t=[],s=[];for(const u of e)t.push(...u.weights),s.push(...u.paths);const r=this.checkManifestAndWeightFiles(e),i=s.map(u=>this.loadWeightsFile(u,r[u]));return Promise.all(i).then(u=>[t,u])}loadWeightsFile(e,t){return new Promise((s,r)=>{const i=new FileReader;i.onload=u=>{const o=u.target.result;s(o)},i.onerror=u=>r(`Failed to weights data from file of path '${e}'.`),i.readAsArrayBuffer(t)})}checkManifestAndWeightFiles(e){const t=[],s=this.weightsFiles.map(i=>Ke(i.name)),r={};for(const i of e)i.paths.forEach(u=>{const o=Ke(u);if(t.indexOf(o)!==-1)throw new Error(`Duplicate file basename found in weights manifest: '${o}'`);if(t.push(o),s.indexOf(o)===-1)throw new Error(`Weight file with basename '${o}' is not provided.`);r[u]=this.weightsFiles[s.indexOf(o)]});if(t.length!==this.weightsFiles.length)throw new Error(`Mismatch in the number of files in weights manifest (${t.length}) and the number of weight files provided (${this.weightsFiles.length}).`);return r}}const Xi=a=>Q().getBool("IS_BROWSER")&&!Array.isArray(a)&&a.startsWith(G.URL_SCHEME)?Zi(a.slice(G.URL_SCHEME.length)):null;ya.registerSaveRouter(Xi);function Zi(a="model"){return new G(a)}function Yi(a){return new Ji(a)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Oe{constructor(e){this.modelArtifacts=e}load(){return this.modelArtifacts}}class At{constructor(e){this.saveHandler=e}save(e){return this.saveHandler(e)}}class Qi{constructor(e){e.load&&(this.load=()=>Promise.resolve(e.load())),e.save&&(this.save=t=>Promise.resolve(e.save(t)))}}function Mi(a,e,t,s){const r=arguments;return new Qi(he(...r))}function he(a,e,t,s){return arguments.length===1?a.modelTopology!=null||a.weightSpecs!=null?new Oe(a):(console.warn("Please call tf.io.fromMemory() with only one argument. The argument should be of type ModelArtifacts. The multi-argument signature of tf.io.fromMemory() has been deprecated and will be removed in a future release."),new Oe({modelTopology:a})):(console.warn("Please call tf.io.fromMemory() with only one argument. The argument should be of type ModelArtifacts. The multi-argument signature of tf.io.fromMemory() has been deprecated and will be removed in a future release."),new Oe({modelTopology:a,weightSpecs:e,weightData:t,trainingConfig:s}))}function eo(a){return new At(a)}function to(a){return new At(a)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Et=Object.freeze(Object.defineProperty({__proto__:null,CompositeArrayBuffer:ft,browserFiles:Yi,browserHTTPRequest:ga,concatenateArrayBuffers:Na,copyModel:ba,decodeWeights:Ta,decodeWeightsStream:Nt,encodeWeights:wa,fromMemory:Mi,fromMemorySync:he,getLoadHandlers:Sa,getModelArtifactsForJSON:yt,getModelArtifactsForJSONSync:bt,getModelArtifactsInfoForJSON:gt,getSaveHandlers:Oa,getWeightSpecs:Tt,http:va,isHTTPScheme:_a,listModels:Aa,loadWeights:Ea,moveModel:Ia,registerLoadRouter:ka,registerSaveRouter:$a,removeModel:Da,weightsLoaderFactory:za,withSaveHandler:eo,withSaveHandlerSync:to},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let H,Ze=!1;function It(a,e=3){if(e>4)throw new Error("Cannot construct Tensor with more than 4 channels from pixels.");if(a==null)throw new Error("pixels passed to tf.browser.fromPixels() can not be null");let t=!1,s=!1,r=!1,i=!1,u=!1,o=!1;if(a.data instanceof Uint8Array)t=!0;else if(typeof ImageData<"u"&&a instanceof ImageData)s=!0;else if(typeof HTMLVideoElement<"u"&&a instanceof HTMLVideoElement)r=!0;else if(typeof HTMLImageElement<"u"&&a instanceof HTMLImageElement)i=!0;else if(a.getContext!=null)u=!0;else if(typeof ImageBitmap<"u"&&a instanceof ImageBitmap)o=!0;else throw new Error(`pixels passed to tf.browser.fromPixels() must be either an HTMLVideoElement, HTMLImageElement, HTMLCanvasElement, ImageData in browser, or OffscreenCanvas, ImageData in webworker or {data: Uint32Array, width: number, height: number}, but was ${a.constructor.name}`);if(wt(Je,v.backendName)!=null){const N={pixels:a},f={numChannels:e};return v.runKernel(Je,N,f)}const[m,l]=r?[a.videoWidth,a.videoHeight]:[a.width,a.height];let c;if(u)c=a.getContext("2d").getImageData(0,0,m,l).data;else if(s||t)c=a.data;else if(i||r||o){if(H==null)if(typeof document>"u")if(typeof OffscreenCanvas<"u"&&typeof OffscreenCanvasRenderingContext2D<"u")H=new OffscreenCanvas(1,1).getContext("2d");else throw new Error("Cannot parse input in current context. Reason: OffscreenCanvas Context2D rendering is not supported.");else H=document.createElement("canvas").getContext("2d",{willReadFrequently:!0});H.canvas.width=m,H.canvas.height=l,H.drawImage(a,0,0,m,l),c=H.getImageData(0,0,m,l).data}let d;if(e===4)d=new Int32Array(c);else{const N=m*l;d=new Int32Array(N*e);for(let f=0;f<N;f++)for(let y=0;y<e;++y)d[f*e+y]=c[f*4+y]}return vt(d,[l,m,e],"int32")}function ao(a){return a!=null&&a.data instanceof Uint8Array}function so(){return typeof window<"u"&&typeof ImageBitmap<"u"&&window.hasOwnProperty("createImageBitmap")}function ro(a){return a!=null&&a.width!==0&&a.height!==0}function no(a){return so()&&!(a instanceof ImageBitmap)&&ro(a)&&!ao(a)}async function io(a,e=3){let t=null;if(Q().getBool("WRAP_TO_IMAGEBITMAP")&&no(a)){let s;try{s=await createImageBitmap(a,{premultiplyAlpha:"none"})}catch{s=null}s!=null&&s.width===a.width&&s.height===a.height?t=s:t=a}else t=a;return It(t,e)}function kt(a){if(a.rank!==2&&a.rank!==3)throw new Error(`toPixels only supports rank 2 or 3 tensors, got rank ${a.rank}.`);const e=a.rank===2?1:a.shape[2];if(e>4||e===2)throw new Error(`toPixels only supports depth of size 1, 3 or 4 but got ${e}`);if(a.dtype!=="float32"&&a.dtype!=="int32")throw new Error(`Unsupported type for toPixels: ${a.dtype}. Please use float32 or int32 tensors.`)}function oo(a){const e=a?.alpha||1;if(e>1||e<0)throw new Error(`Alpha value ${e} is suppoed to be in range [0 - 1].`)}async function uo(a,e){let t=g(a,"img","toPixels");if(!(a instanceof Z)){const m=t;t=Fe(m,"int32"),m.dispose()}kt(t);const[s,r]=t.shape.slice(0,2),i=t.rank===2?1:t.shape[2],u=await t.data(),o=t.dtype==="float32"?255:1,p=new Uint8ClampedArray(r*s*4);for(let m=0;m<s*r;++m){const l=[0,0,0,255];for(let d=0;d<i;d++){const h=u[m*i+d];if(t.dtype==="float32"){if(h<0||h>1)throw new Error(`Tensor values for a float32 Tensor must be in the range [0 - 1] but encountered ${h}.`)}else if(t.dtype==="int32"&&(h<0||h>255))throw new Error(`Tensor values for a int32 Tensor must be in the range [0 - 255] but encountered ${h}.`);i===1?(l[0]=h*o,l[1]=h*o,l[2]=h*o):l[d]=h*o}const c=m*4;p[c+0]=Math.round(l[0]),p[c+1]=Math.round(l[1]),p[c+2]=Math.round(l[2]),p[c+3]=Math.round(l[3])}if(e!=null){Ze||wt(St,v.backendName)!=null&&(console.warn("tf.browser.toPixels is not efficient to draw tensor on canvas. Please try tf.browser.draw instead."),Ze=!0),e.width=r,e.height=s;const m=e.getContext("2d"),l=new ImageData(p,r,s);m.putImageData(l,0,0)}return t!==a&&t.dispose(),p}function po(a,e,t){let s=g(a,"img","draw");if(!(a instanceof Z)){const u=s;s=Fe(u,"int32"),u.dispose()}kt(s),oo(t?.imageOptions);const r={image:s},i={canvas:e,options:t};v.runKernel(St,r,i)}const mo=T({fromPixels_:It}),ju=Object.freeze(Object.defineProperty({__proto__:null,draw:po,fromPixels:mo,fromPixelsAsync:io,toPixels:uo},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const xe={};function xu(a,e){const t={tfOpName:a,category:"custom",inputs:[],attrs:[],customExecutor:e};xe[a]=t}function $t(a){return xe[a]}function Ru(a){delete xe[a]}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function n(a,e,t,s,r){const i=e.inputParams[a];if(i&&i.inputIndexStart!==void 0){const o=i.inputIndexStart,p=i.inputIndexEnd===0?void 0:i.inputIndexEnd===void 0?o+1:i.inputIndexEnd,m=o<0?e.inputNames.length+o:o;if(i.type==="tensor")return A(e.inputNames[m],t,s,r);if(i.type==="tensors"){const d=e.inputs.slice(o,p);return e.inputNames.slice(o,p).filter((N,f)=>{var y;return((y=d[f])===null||y===void 0?void 0:y.op)!=="NoOp"}).map(N=>A(N,t,s,r))}const l=A(e.inputNames[m],t,s,r),c=l.dataSync();return i.type==="number"?c[0]:La(l.shape,c)}const u=e.attrParams[a];return u&&u.value}function A(a,e,t,s){const[r,i]=$(a,t);if(s!=null){const o=s.getHashTableHandleByName(r);if(o!=null)return o}const u=t.currentContextIds.find(o=>!!e[ye(r,o)]);return u!==void 0?e[ye(r,u)][i]:void 0}function Ye(a,e,t){return e[ye(a,t.currentContextId)]}function j(a,e){const[t,s,r]=$(a,e);return[ye(t,e&&e.currentContextId),s,r]}function ye(a,e){return e?`${a}-${e}`:a}function $(a,e){if(a==="")return["",0,void 0];const t=e!=null&&e.parseNodeNameCache!=null;if(t){const i=e.parseNodeNameCache.get(a);if(i!=null)return i}const s=a.split(":");let r;if(s.length===1)r=[a,0,void 0];else{const i=s[0],u=s.length===3?s[1]:void 0,o=Number(s[s.length-1]);r=[i,o,u]}return t&&e.parseNodeNameCache.set(a,r),r}function le(a,e,t){let s=n("pad",a,e,t);if(s==="explicit"){s=n("explicitPaddings",a,e,t);const r=[[0,0],[0,0],[0,0],[0,0]];for(let i=0;i<4;i++)r[i][0]=s[i*2],r[i][1]=s[i*2+1];return r}return s}function x(a){return a.kept?a:Ot(a)}/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const lo=[{tfOpName:"Add",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"AddV2",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"AddN",category:"arithmetic",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}]},{tfOpName:"BiasAdd",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"Sub",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"RealDiv",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Div",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"DivNoNan",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"FloorDiv",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Mul",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Maximum",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Minimum",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Pow",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"SquaredDifference",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Mod",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"FloorMod",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}],co=Object.freeze(Object.defineProperty({__proto__:null,json:lo},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ho=[{tfOpName:"Abs",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Acos",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Asin",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Atan",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Atan2",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"y",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Ceil",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ClipByValue",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"clipValueMin",type:"number"},{start:2,name:"clipValueMax",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Complex",category:"basic_math",inputs:[{start:0,name:"real",type:"tensor"},{start:1,name:"imag",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ComplexAbs",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Cos",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Cosh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Elu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Exp",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Floor",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Log",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Imag",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"Tout",name:"outputType",type:"dtype",notSupported:!0}]},{tfOpName:"Neg",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Real",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"Tout",name:"outputType",type:"dtype",notSupported:!0}]},{tfOpName:"Prelu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"alpha",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Relu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Relu6",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Selu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sigmoid",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sin",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sinh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sqrt",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Rsqrt",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Square",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Tan",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Tanh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sign",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Round",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Expm1",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Log1p",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Reciprocal",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Softplus",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Asinh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Acosh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Atanh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Erf",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LeakyRelu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"alpha",name:"alpha",type:"number",defaultValue:.2},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"IsNan",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"IsFinite",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"IsInf",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}],yo=Object.freeze(Object.defineProperty({__proto__:null,json:ho},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const fo=[{tfOpName:"EmptyTensorList",category:"control",inputs:[{start:0,name:"elementShape",type:"shape"},{start:1,name:"maxNumElements",type:"number"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"LoopCond",category:"control",inputs:[{start:0,name:"pred",type:"tensor"}]},{tfOpName:"Switch",category:"control",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"pred",type:"tensor"}]},{tfOpName:"Merge",category:"control",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}]},{tfOpName:"Enter",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"frame_name",name:"frameName",type:"string"},{tfName:"is_constant",name:"isConstant",type:"bool"}]},{tfOpName:"Exit",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"NextIteration",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"TensorArrayV3",category:"control",inputs:[{start:0,name:"size",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"element_shape",name:"elementShape",type:"shape"},{tfName:"dynamic_size",name:"dynamicSize",type:"bool"},{tfName:"clear_after_read",name:"clearAfterRead",type:"bool"},{tfName:"identical_element_shapes",name:"identicalElementShapes",type:"bool"},{tfName:"tensor_array_name",name:"name",type:"string"}]},{tfOpName:"TensorArrayWriteV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"tensor",type:"tensor"},{start:3,name:"flowIn",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"TensorArrayReadV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"flowIn",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"TensorArrayGatherV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"flowIn",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"element_shape",name:"elementShape",type:"shape"}]},{tfOpName:"TensorArrayScatterV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"tensor",type:"tensor"},{start:3,name:"flowIn",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"TensorArrayConcatV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"flowIn",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"element_shape_except0",name:"elementShapeExcept0",type:"shape",notSupported:!0}]},{tfOpName:"TensorArraySplitV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"tensor",type:"tensor"},{start:2,name:"lengths",type:"number[]"},{start:3,name:"flowIn",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"TensorArraySizeV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"flowIn",type:"number"}]},{tfOpName:"TensorArrayCloseV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"}]},{tfOpName:"StatelessIf",category:"control",inputs:[{start:0,name:"cond",type:"tensor"},{start:1,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"then_branch",name:"thenBranch",type:"func"},{tfName:"else_branch",name:"elseBranch",type:"func"}]},{tfOpName:"If",category:"control",inputs:[{start:0,name:"cond",type:"tensor"},{start:1,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"then_branch",name:"thenBranch",type:"func"},{tfName:"else_branch",name:"elseBranch",type:"func"}]},{tfOpName:"StatelessWhile",category:"control",inputs:[{start:0,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"cond",name:"cond",type:"func"},{tfName:"body",name:"body",type:"func"}]},{tfOpName:"While",category:"control",inputs:[{start:0,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"cond",name:"cond",type:"func"},{tfName:"body",name:"body",type:"func"}]},{tfOpName:"TensorListScatter",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListScatterV2",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"elementShape",type:"shape"},{start:3,name:"numElements",type:"number"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListGather",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListGetItem",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListSetItem",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"tensor",type:"tensor"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListReserve",category:"control",inputs:[{start:0,name:"elementShape",type:"shape"},{start:1,name:"numElements",type:"number"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListFromTensor",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListStack",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"},{tfName:"num_elements",name:"numElements",type:"dtype"}]},{tfOpName:"TensorListSplit",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"elementShape",type:"shape"},{start:2,name:"lengths",type:"number[]"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListConcat",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"}],attrs:[{tfName:"element_shape",name:"elementShape",type:"shape"},{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListConcatV2",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"}],attrs:[{tfName:"element_shape",name:"elementShape",type:"shape"},{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListPopBack",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListPushBack",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"tensor",type:"tensor"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListLength",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"}]},{tfOpName:"TensorListResize",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"size",type:"number"}]}],go=Object.freeze(Object.defineProperty({__proto__:null,json:fo},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const No=[{tfOpName:"AvgPool",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MaxPool",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[],notSupported:!0},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MaxPoolWithArgmax",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"include_batch_in_index",name:"includeBatchInIndex",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"AvgPool3D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MaxPool3D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Conv1D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"stride",name:"stride",type:"number"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NWC"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"dilation",name:"dilation",type:"number",defaultValue:1}]},{tfOpName:"Conv2D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"useCudnnOnGpu",name:"useCudnnOnGpu",type:"bool"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"_FusedConv2D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"},{start:2,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"num_args",name:"numArgs",type:"number"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"use_cudnn_on_gpu",name:"useCudnnOnGpu",type:"bool",defaultValue:!0},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"dilations",name:"dilations",type:"number[]",defaultValue:[1,1,1,1]},{tfName:"fused_ops",name:"fusedOps",type:"string[]",defaultValue:[]},{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:1e-4},{tfName:"leakyrelu_alpha",name:"leakyreluAlpha",type:"number",defaultValue:.2}]},{tfOpName:"Conv2DBackpropInput",category:"convolution",inputs:[{start:2,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"},{start:0,name:"outputShape",type:"number[]"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]",notSupported:!0}]},{tfOpName:"DepthwiseConv2d",category:"convolution",inputs:[{start:0,name:"input",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"DepthwiseConv2dNative",category:"convolution",inputs:[{start:0,name:"input",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"FusedDepthwiseConv2dNative",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"},{start:2,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"num_args",name:"numArgs",type:"number"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"dilations",name:"dilations",type:"number[]",defaultValue:[1,1,1,1]},{tfName:"fused_ops",name:"fusedOps",type:"string[]",defaultValue:[]},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]}]},{tfOpName:"Conv3D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"Dilation2D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"rates",name:"dilations",type:"number[]"},{tfName:"padding",name:"pad",type:"string"}]}],bo=Object.freeze(Object.defineProperty({__proto__:null,json:No},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const To=[{tfOpName:"Fill",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"},{start:1,name:"value",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"LinSpace",category:"creation",inputs:[{start:0,name:"start",type:"number"},{start:1,name:"stop",type:"number"},{start:2,name:"num",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"OneHot",category:"creation",inputs:[{start:0,name:"indices",type:"tensor"},{start:1,name:"depth",type:"number"},{start:2,name:"onValue",type:"number",defaultValue:1},{start:3,name:"offValue",type:"number",defaultValue:0}],attrs:[{tfName:"axis",name:"axis",type:"number",notSupported:!0},{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"Ones",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"OnesLike",category:"creation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"}]},{tfOpName:"RandomStandardNormal",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"seed",name:"seed",type:"number",defaultValue:0},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0},{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"T",name:"T",type:"number",notSupported:!0}]},{tfOpName:"RandomUniform",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"minval",name:"minval",type:"number",defaultValue:0},{tfName:"maxval",name:"maxval",type:"number",defaultValue:1},{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"seed",name:"seed",type:"number",defaultValue:0},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0},{tfName:"T",name:"T",type:"number",notSupported:!0}]},{tfOpName:"RandomUniformInt",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"minval",name:"minval",type:"number"},{tfName:"maxval",name:"maxval",type:"number"},{tfName:"seed",name:"seed",type:"number",defaultValue:0},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0}]},{tfOpName:"Range",category:"creation",inputs:[{start:0,name:"start",type:"number"},{start:1,name:"stop",type:"number"},{start:2,name:"step",type:"number",defaultValue:0}],attrs:[{tfName:"Tidx",name:"dtype",type:"dtype"}]},{tfOpName:"TruncatedNormal",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"means",name:"mean",type:"number",defaultValue:0},{tfName:"stddev",name:"stdDev",type:"number",defaultValue:1},{tfName:"seed",name:"seed",type:"number"},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0},{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"T",name:"T",type:"number",notSupported:!0}]},{tfOpName:"Zeros",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"ZerosLike",category:"creation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"Multinomial",category:"creation",inputs:[{start:0,name:"logits",type:"tensor"},{start:1,name:"numSamples",type:"number"}],attrs:[{tfName:"seed",name:"seed",type:"number"},{tfName:"seed2",name:"seed2",type:"number"},{tfName:"T",name:"dtype",type:"dtype"},{tfName:"output_dtype",name:"output_dtype",type:"dtype"}]}],wo=Object.freeze(Object.defineProperty({__proto__:null,json:To},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const So=[{tfOpName:"NonMaxSuppressionV2",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"}]},{tfOpName:"NonMaxSuppressionV3",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"},{start:4,name:"scoreThreshold",type:"number"}]},{tfOpName:"NonMaxSuppressionV4",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"},{start:4,name:"scoreThreshold",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"T_threshold",name:"threshold",type:"dtype",notSupported:!0},{tfName:"pad_to_max_output_size",name:"padToMaxOutputSize",type:"bool"}]},{tfOpName:"NonMaxSuppressionV5",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"},{start:4,name:"scoreThreshold",type:"number"},{start:5,name:"softNmsSigma",type:"number"}]},{tfOpName:"Where",category:"dynamic",inputs:[{start:0,name:"condition",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ListDiff",category:"dynamic",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"y",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}],Oo=Object.freeze(Object.defineProperty({__proto__:null,json:So},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const vo=[{tfOpName:"LowerBound",category:"evaluation",inputs:[{start:0,name:"sortedSequence",type:"tensor"},{start:1,name:"values",type:"tensor"}]},{tfOpName:"TopKV2",category:"evaluation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"k",type:"number"}],attrs:[{tfName:"sorted",name:"sorted",type:"bool"}]},{tfOpName:"UpperBound",category:"evaluation",inputs:[{start:0,name:"sortedSequence",type:"tensor"},{start:1,name:"values",type:"tensor"}]},{tfOpName:"Unique",category:"evaluation",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"UniqueV2",category:"evaluation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]}],_o=Object.freeze(Object.defineProperty({__proto__:null,json:vo},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ao=[{tfOpName:"PlaceholderWithDefault",category:"graph",inputs:[{start:0,name:"default",type:"tensor"}],attrs:[{tfName:"shape",name:"shape",type:"shape"},{tfName:"dtype",name:"dtype",type:"dtype"}]},{tfOpName:"Placeholder",category:"graph",attrs:[{tfName:"shape",name:"shape",type:"shape"},{tfName:"dtype",name:"dtype",type:"dtype"}]},{tfOpName:"Const",category:"graph"},{tfOpName:"Identity",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"IdentityN",category:"graph",inputs:[{start:0,end:0,name:"x",type:"tensors"}]},{tfOpName:"Snapshot",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"Rank",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"Size",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"Shape",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"ShapeN",category:"graph",inputs:[{start:0,end:0,name:"x",type:"tensors"}]},{tfOpName:"Print",category:"graph",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"data",type:"tensors"}],attrs:[{tfName:"message",name:"message",type:"string"},{tfName:"first_n",name:"firstN",type:"number",notSupported:!0},{tfName:"summarize",name:"summarize",type:"number",defaultValue:3}]},{tfOpName:"NoOp",category:"graph",inputs:[]},{tfOpName:"StopGradient",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"FakeQuantWithMinMaxVars",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"min",name:"min",type:"number"},{tfName:"max",name:"max",type:"number"}]}],Eo=Object.freeze(Object.defineProperty({__proto__:null,json:Ao},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Io=[{tfOpName:"HashTable",category:"hash_table",inputs:[],attrs:[{tfName:"shared_name",name:"sharedName",type:"string"},{tfName:"use_node_name_sharing",name:"useNodeNameSharing",type:"bool"},{tfName:"key_dtype",name:"keyDType",type:"dtype"},{tfName:"value_dtype",name:"valueDType",type:"dtype"}]},{tfOpName:"HashTableV2",category:"hash_table",inputs:[],attrs:[{tfName:"shared_name",name:"sharedName",type:"string"},{tfName:"use_node_name_sharing",name:"useNodeNameSharing",type:"bool"},{tfName:"key_dtype",name:"keyDType",type:"dtype"},{tfName:"value_dtype",name:"valueDType",type:"dtype"}]},{tfOpName:"LookupTableImport",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableImportV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableFind",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableFindV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableSize",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"}]},{tfOpName:"LookupTableSizeV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"}]},{tfOpName:"InitializeTable",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}]},{tfOpName:"InitializeTableV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}]}],ko=Object.freeze(Object.defineProperty({__proto__:null,json:Io},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const $o=[{tfOpName:"ResizeBilinear",category:"image",inputs:[{start:0,name:"images",type:"tensor"},{start:1,name:"size",type:"number[]"}],attrs:[{tfName:"align_corners",name:"alignCorners",type:"bool"},{tfName:"half_pixel_centers",name:"halfPixelCenters",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ResizeNearestNeighbor",category:"image",inputs:[{start:0,name:"images",type:"tensor"},{start:1,name:"size",type:"number[]"}],attrs:[{tfName:"align_corners",name:"alignCorners",type:"bool"},{tfName:"half_pixel_centers",name:"halfPixelCenters",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"CropAndResize",category:"image",inputs:[{start:0,name:"image",type:"tensor"},{start:1,name:"boxes",type:"tensor"},{start:2,name:"boxInd",type:"tensor"},{start:3,name:"cropSize",type:"number[]"}],attrs:[{tfName:"method",name:"method",type:"string"},{tfName:"extrapolation_value",name:"extrapolationValue",type:"number"}]},{tfOpName:"ImageProjectiveTransformV3",category:"image",inputs:[{start:0,name:"images",type:"tensor"},{start:1,name:"transforms",type:"tensor"},{start:2,name:"outputShape",type:"number[]"},{start:3,name:"fillValue",type:"number"}],attrs:[{tfName:"interpolation",name:"interpolation",type:"string"},{tfName:"fill_mode",name:"fillMode",type:"string"}]}],Do=Object.freeze(Object.defineProperty({__proto__:null,json:$o},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const zo=[{tfOpName:"Equal",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"NotEqual",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Greater",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"GreaterEqual",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Less",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LessEqual",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LogicalAnd",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LogicalNot",category:"logical",inputs:[{start:0,name:"a",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LogicalOr",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Select",category:"logical",inputs:[{start:0,name:"condition",type:"tensor"},{start:1,name:"a",type:"tensor"},{start:2,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"SelectV2",category:"logical",inputs:[{start:0,name:"condition",type:"tensor"},{start:1,name:"a",type:"tensor"},{start:2,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"BitwiseAnd",category:"logical",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"y",type:"tensor"}]}],Lo=Object.freeze(Object.defineProperty({__proto__:null,json:zo},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Co=[{tfOpName:"_FusedMatMul",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"},{start:2,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"num_args",name:"numArgs",type:"number"},{tfName:"fused_ops",name:"fusedOps",type:"string[]",defaultValue:[]},{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:1e-4},{tfName:"transpose_a",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"transpose_b",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"leakyrelu_alpha",name:"leakyreluAlpha",type:"number",defaultValue:.2},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MatMul",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"transpose_a",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"transpose_b",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"BatchMatMul",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"adj_x",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"adj_y",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"BatchMatMulV2",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"adj_x",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"adj_y",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Transpose",category:"matrices",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"perm",type:"number[]"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Einsum",category:"matrices",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}],attrs:[{tfName:"equation",name:"equation",type:"string"},{tfName:"N",name:"n",type:"number",defaultValue:2},{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"MatrixBandPart",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"numLower",type:"tensor"},{start:1,name:"numUpper",type:"tensor"}]}],Vo=Object.freeze(Object.defineProperty({__proto__:null,json:Co},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Po=[{tfOpName:"EuclideanNorm",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool",defaultValue:!1}]},{tfOpName:"FusedBatchNorm",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"scale",type:"tensor"},{start:2,name:"offset",type:"tensor"},{start:3,name:"mean",type:"tensor"},{start:4,name:"variance",type:"tensor"}],attrs:[{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:.001},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"FusedBatchNormV2",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"scale",type:"tensor"},{start:2,name:"offset",type:"tensor"},{start:3,name:"mean",type:"tensor"},{start:4,name:"variance",type:"tensor"}],attrs:[{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:.001},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"FusedBatchNormV3",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"scale",type:"tensor"},{start:2,name:"offset",type:"tensor"},{start:3,name:"mean",type:"tensor"},{start:4,name:"variance",type:"tensor"}],attrs:[{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:.001},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"LRN",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"depth_radius",name:"radius",type:"number",defaultValue:5},{tfName:"bias",name:"bias",type:"number",defaultValue:1},{tfName:"alpha",name:"alpha",type:"number",defaultValue:1},{tfName:"beta",name:"beta",type:"number",defaultValue:.5}]},{tfOpName:"Softmax",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"LogSoftmax",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"}]}],Fo=Object.freeze(Object.defineProperty({__proto__:null,json:Po},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const jo=[{tfOpName:"Bincount",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"size",type:"number"},{start:2,name:"weights",type:"tensor"}]},{tfOpName:"DenseBincount",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"size",type:"number"},{start:2,name:"weights",type:"tensor"}],attrs:[{tfName:"binary_output",name:"binaryOutput",type:"bool"}]},{tfOpName:"Max",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Mean",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Min",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Sum",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"All",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Any",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"ArgMax",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]},{tfOpName:"ArgMin",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]},{tfOpName:"Prod",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Cumprod",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}],attrs:[{tfName:"exclusive",name:"exclusive",type:"bool"},{tfName:"reverse",name:"reverse",type:"bool"}]},{tfOpName:"Cumsum",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}],attrs:[{tfName:"exclusive",name:"exclusive",type:"bool"},{tfName:"reverse",name:"reverse",type:"bool"}]}],xo=Object.freeze(Object.defineProperty({__proto__:null,json:jo},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ro=[{tfOpName:"ConcatV2",category:"slice_join",inputs:[{start:0,end:-1,name:"tensors",type:"tensors"},{start:-1,name:"axis",type:"number"}],attrs:[{tfName:"N",name:"n",type:"number",defaultValue:2}]},{tfOpName:"Concat",category:"slice_join",inputs:[{start:1,end:0,name:"tensors",type:"tensors"},{start:0,name:"axis",type:"number"}],attrs:[{tfName:"N",name:"n",type:"number",defaultValue:2}]},{tfOpName:"GatherV2",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"axis",type:"number",defaultValue:0}],attrs:[{tfName:"batch_dims",name:"batchDims",type:"number",defaultValue:0}]},{tfOpName:"Gather",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"indices",type:"tensor"}],attrs:[{tfName:"validate_indices",name:"validateIndices",type:"bool",notSupported:!0}]},{tfOpName:"Reverse",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"dims",type:"bool[]"}]},{tfOpName:"ReverseV2",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}]},{tfOpName:"Slice",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"begin",type:"number[]"},{start:2,name:"size",type:"number[]"}]},{tfOpName:"StridedSlice",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"begin",type:"number[]"},{start:2,name:"end",type:"number[]"},{start:3,name:"strides",type:"number[]"}],attrs:[{tfName:"begin_mask",name:"beginMask",type:"number",defaultValue:0},{tfName:"end_mask",name:"endMask",type:"number",defaultValue:0},{tfName:"new_axis_mask",name:"newAxisMask",type:"number",defaultValue:0},{tfName:"ellipsis_mask",name:"ellipsisMask",type:"number",defaultValue:0},{tfName:"shrink_axis_mask",name:"shrinkAxisMask",type:"number",defaultValue:0}]},{tfOpName:"Pack",category:"slice_join",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}],attrs:[{tfName:"axis",name:"axis",type:"number",defaultValue:0}]},{tfOpName:"Unpack",category:"slice_join",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"axis",name:"axis",type:"number",defaultValue:0},{tfName:"num",name:"num",type:"number",defaultValue:0,notSupported:!0}]},{tfOpName:"Tile",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"reps",type:"number[]"}]},{tfOpName:"Split",category:"slice_join",inputs:[{start:0,name:"axis",type:"number",defaultValue:0},{start:1,name:"x",type:"tensor"}],attrs:[{tfName:"num_split",name:"numOrSizeSplits",type:"number",defaultValue:1}]},{tfOpName:"SplitV",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"numOrSizeSplits",type:"number[]"},{start:2,name:"axis",type:"number",defaultValue:0}]},{tfOpName:"ScatterNd",category:"slice_join",inputs:[{start:0,name:"indices",type:"tensor"},{start:1,name:"values",type:"tensor"},{start:2,name:"shape",type:"number[]"}]},{tfOpName:"GatherNd",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"indices",type:"tensor"}]},{tfOpName:"SparseToDense",category:"slice_join",inputs:[{start:0,name:"sparseIndices",type:"tensor"},{start:1,name:"outputShape",type:"number[]"},{start:2,name:"sparseValues",type:"tensor"},{start:3,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"validate_indices",name:"validateIndices",type:"bool",defaultValue:!1,notSupported:!0}]},{tfOpName:"TensorScatterUpdate",category:"slice_join",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"values",type:"tensor"}]}],Bo=Object.freeze(Object.defineProperty({__proto__:null,json:Ro},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ho=[{tfOpName:"SparseFillEmptyRows",category:"sparse",inputs:[{start:0,name:"indices",type:"tensor"},{start:1,name:"values",type:"tensor"},{start:2,name:"denseShape",type:"tensor"},{start:3,name:"defaultValue",type:"tensor"}]},{tfOpName:"SparseReshape",category:"sparse",inputs:[{start:0,name:"inputIndices",type:"tensor"},{start:1,name:"inputShape",type:"tensor"},{start:2,name:"newShape",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"SparseSegmentMean",category:"sparse",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"segmentIds",type:"tensor"}]},{tfOpName:"SparseSegmentSum",category:"sparse",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"segmentIds",type:"tensor"}]}],Wo=Object.freeze(Object.defineProperty({__proto__:null,json:Ho},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const qo=[{tfOpName:"FFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"IFFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"RFFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"fft_length",type:"number",notSupported:!0}]},{tfOpName:"IRFFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"fft_length",type:"number",notSupported:!0}]}],Uo=Object.freeze(Object.defineProperty({__proto__:null,json:qo},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Go=[{tfOpName:"StaticRegexReplace",category:"string",inputs:[{start:0,name:"input",type:"tensor"}],attrs:[{tfName:"pattern",name:"pattern",type:"string"},{tfName:"rewrite",name:"rewrite",type:"string"},{tfName:"replace_global",name:"replaceGlobal",type:"bool"}]},{tfOpName:"StringNGrams",category:"string",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"dataSplits",type:"tensor"}],attrs:[{tfName:"separator",name:"separator",type:"string"},{tfName:"ngram_widths",name:"nGramWidths",type:"number[]"},{tfName:"left_pad",name:"leftPad",type:"string"},{tfName:"right_pad",name:"rightPad",type:"string"},{tfName:"pad_width",name:"padWidth",type:"number"},{tfName:"preserve_short_sequences",name:"preserveShortSequences",type:"bool"}],outputs:["ngrams","ngrams_splits"]},{tfOpName:"StringSplit",category:"string",inputs:[{start:0,name:"input",type:"tensor"},{start:1,name:"delimiter",type:"tensor"}],attrs:[{tfName:"skip_empty",name:"skipEmpty",type:"bool"}],outputs:["indices","values","shape"]},{tfOpName:"StringToHashBucketFast",category:"string",inputs:[{start:0,name:"input",type:"tensor"}],attrs:[{tfName:"num_buckets",name:"numBuckets",type:"number"}]}],Ko=Object.freeze(Object.defineProperty({__proto__:null,json:Go},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Jo=[{tfOpName:"Cast",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"SrcT",name:"sdtype",type:"dtype",notSupported:!0},{tfName:"DstT",name:"dtype",type:"dtype"}]},{tfOpName:"ExpandDims",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]},{tfOpName:"MirrorPad",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"padding",type:"number[]"}],attrs:[{tfName:"mode",name:"mode",type:"string"}]},{tfOpName:"Pad",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"padding",type:"number[]"}],attrs:[{tfName:"constant_value",name:"constantValue",type:"number",defaultValue:0}]},{tfOpName:"PadV2",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"padding",type:"number[]"},{start:2,name:"constantValue",type:"number",defaultValue:0}]},{tfOpName:"Reshape",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"shape",type:"number[]"}]},{tfOpName:"EnsureShape",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"shape",type:"number[]"}]},{tfOpName:"Squeeze",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"axis",tfDeprecatedName:"squeeze_dims",name:"axis",type:"number[]"}]},{tfOpName:"SpaceToBatchND",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"blockShape",type:"number[]"},{start:2,name:"paddings",type:"number[]"}]},{tfOpName:"BatchToSpaceND",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"blockShape",type:"number[]"},{start:2,name:"crops",type:"number[]"}]},{tfOpName:"DepthToSpace",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"block_size",name:"blockSize",type:"number"},{tfName:"data_format",name:"dataFormat",type:"string"}]},{tfOpName:"BroadcastTo",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"shape",type:"number[]"}],attrs:[]},{tfOpName:"BroadcastArgs",category:"transformation",inputs:[{start:0,name:"s0",type:"tensor"},{start:1,name:"s1",type:"tensor"}],attrs:[]}],Xo=Object.freeze(Object.defineProperty({__proto__:null,json:Jo},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Qe{static get Instance(){return this._instance||(this._instance=new this)}constructor(){const e=[co,yo,go,bo,wo,Oo,_o,Eo,ko,Do,Lo,Vo,Fo,xo,Bo,Wo,Uo,Ko,Xo],t=[].concat(...e.map(s=>s.json));this.opMappers=t.reduce((s,r)=>(s[r.tfOpName]=r,s),{})}transformGraph(e,t={}){const s=e.node,r=[],i=[],u=[],o=s.reduce((f,y)=>(f[y.name]=this.mapNode(y),y.op.startsWith("Placeholder")?r.push(f[y.name]):y.op==="Const"?i.push(f[y.name]):(y.input==null||y.input.length===0)&&u.push(f[y.name]),f),{});let p=[];const m=[];let l={},c={};t!=null&&(l=this.mapSignatureEntries(t.inputs),c=this.mapSignatureEntries(t.outputs));const d=Object.keys(o);d.forEach(f=>{const y=o[f];y.inputNames.forEach((b,_)=>{const[k,,w]=j(b),I=o[k];if(I.outputs!=null){const D=I.outputs.indexOf(w);if(D!==-1){const z=`${k}:${D}`;y.inputNames[_]=z}}y.inputs.push(I),I.children.push(y)})}),Object.keys(c).length===0?d.forEach(f=>{const y=o[f];y.children.length===0&&m.push(y)}):Object.keys(c).forEach(f=>{const[y]=j(f),b=o[y];b!=null&&(b.signatureKey=c[f],m.push(b))}),Object.keys(l).length>0?Object.keys(l).forEach(f=>{const[y]=j(f),b=o[y];b&&(b.signatureKey=l[f],p.push(b))}):p=r;let h={};e.library!=null&&e.library.function!=null&&(h=e.library.function.reduce((f,y)=>(f[y.signature.name]=this.mapFunction(y),f),{}));const N={nodes:o,inputs:p,outputs:m,weights:i,placeholders:r,signature:t,functions:h};return u.length>0&&(N.initNodes=u),N}mapSignatureEntries(e){return Object.keys(e||{}).reduce((t,s)=>(t[e[s].name]=s,t),{})}mapNode(e){const t=$t(e.op)||this.opMappers[e.op]||{};e.attr==null&&(e.attr={});const s={name:e.name,op:e.op,category:t.category,inputNames:(e.input||[]).map(r=>r.startsWith("^")?r.slice(1):r),inputs:[],children:[],inputParams:{},attrParams:{},rawAttrs:e.attr,outputs:t.outputs};return t.inputs!=null&&(s.inputParams=t.inputs.reduce((r,i)=>(r[i.name]={type:i.type,inputIndexStart:i.start,inputIndexEnd:i.end},r),{})),t.attrs!=null&&(s.attrParams=t.attrs.reduce((r,i)=>{const u=i.type;let o;switch(i.type){case"string":o=Ae(e.attr,i.tfName,i.defaultValue),o===void 0&&i.tfDeprecatedName&&(o=Ae(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"string[]":o=Le(e.attr,i.tfName,i.defaultValue),o===void 0&&i.tfDeprecatedName&&(o=Le(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"number":o=Ie(e.attr,i.tfName,i.defaultValue||0),o===void 0&&i.tfDeprecatedName&&(o=Ie(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"number[]":o=ze(e.attr,i.tfName,i.defaultValue),o===void 0&&i.tfDeprecatedName&&(o=ze(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"bool":o=Ee(e.attr,i.tfName,i.defaultValue),o===void 0&&i.tfDeprecatedName&&(o=Ee(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"bool[]":o=Ve(e.attr,i.tfName,i.defaultValue),o===void 0&&i.tfDeprecatedName&&(o=Ve(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"shape":o=De(e.attr,i.tfName,i.defaultValue),o===void 0&&i.tfDeprecatedName&&(o=De(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"shape[]":o=Ce(e.attr,i.tfName,i.defaultValue),o===void 0&&i.tfDeprecatedName&&(o=Ce(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"dtype":o=ke(e.attr,i.tfName,i.defaultValue),o===void 0&&i.tfDeprecatedName&&(o=ke(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"dtype[]":o=$e(e.attr,i.tfName,i.defaultValue),o===void 0&&i.tfDeprecatedName&&(o=$e(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"func":o=Me(e.attr,i.tfName,i.defaultValue),o===void 0&&i.tfDeprecatedName&&(o=Me(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"tensor":case"tensors":break;default:throw new Error(`Unsupported param type: ${i.type} for op: ${e.op}`)}return r[i.name]={value:o,type:u},r},{})),s}mapFunction(e){const t=e.nodeDef,s=[],r=[];let i={};t!=null&&(i=t.reduce((c,d)=>(c[d.name]=this.mapNode(d),d.op==="Const"&&r.push(c[d.name]),c),{}));const u=[],o=[];e.signature.inputArg.forEach(c=>{const[d]=j(c.name),h={name:d,op:"Placeholder",inputs:[],inputNames:[],category:"graph",inputParams:{},attrParams:{dtype:{value:Re(c.type),type:"dtype"}},children:[]};h.signatureKey=c.name,u.push(h),i[d]=h}),Object.keys(i).forEach(c=>{const d=i[c];d.inputNames.forEach((h,N)=>{const[f,,y]=j(h),b=i[f];if(b.outputs!=null){const _=b.outputs.indexOf(y);if(_!==-1){const k=`${f}:${_}`;d.inputNames[N]=k}}d.inputs.push(b),b.children.push(d)})});const m=e.ret;e.signature.outputArg.forEach(c=>{const[d,h]=j(m[c.name]),N=i[d];N!=null&&(N.defaultOutput=h,o.push(N))});const l=this.mapArgsToSignature(e);return{nodes:i,inputs:u,outputs:o,weights:r,placeholders:s,signature:l}}mapArgsToSignature(e){return{methodName:e.signature.name,inputs:e.signature.inputArg.reduce((t,s)=>(t[s.name]=this.mapArgToTensorInfo(s),t),{}),outputs:e.signature.outputArg.reduce((t,s)=>(t[s.name]=this.mapArgToTensorInfo(s,e.ret),t),{})}}mapArgToTensorInfo(e,t){let s=e.name;return t!=null&&(s=t[s]),{name:s,dtype:e.type}}}function Zo(a){const e=Q().global;if(typeof e.atob<"u")return e.atob(a);if(typeof Buffer<"u")return new Buffer(a,"base64").toString();throw new Error("Unable to decode base64 in this environment. Missing built-in atob() or Buffer()")}function Dt(a,e){const t=Array.isArray(a)?String.fromCharCode.apply(null,a):Zo(a);return e?t:t.toLowerCase()}function Ae(a,e,t,s=!1){const r=a[e];return r!=null?Dt(r.s,s):t}function Ee(a,e,t){const s=a[e];return s?s.b:t}function Ie(a,e,t){const s=a[e]||{},r=s.i!=null?s.i:s.f!=null?s.f:t;return typeof r=="number"?r:parseInt(r,10)}function Re(a){switch(typeof a=="string"&&(a=L[a]),a){case L.DT_FLOAT:case L.DT_HALF:return"float32";case L.DT_INT32:case L.DT_INT64:case L.DT_INT8:case L.DT_UINT8:return"int32";case L.DT_BOOL:return"bool";case L.DT_DOUBLE:return"float32";case L.DT_STRING:return"string";case L.DT_COMPLEX64:case L.DT_COMPLEX128:return"complex64";default:return null}}function Me(a,e,t){const s=a[e];return s&&s.func?s.func.name:t}function ke(a,e,t){const s=a[e];return s&&s.type?Re(s.type):t}function $e(a,e,t){const s=a[e];return s&&s.list&&s.list.type?s.list.type.map(r=>Re(r)):t}function zt(a){if(!a.unknownRank)return a.dim!=null?a.dim.map(e=>typeof e.size=="number"?e.size:parseInt(e.size,10)):[]}function De(a,e,t){const s=a[e];return s&&s.shape?zt(s.shape):t}function ze(a,e,t){const s=a[e];return s?((s.list.f&&s.list.f.length?s.list.f:s.list.i)||[]).map(r=>typeof r=="number"?r:parseInt(r,10)):t}function Le(a,e,t,s=!1){const r=a[e];return r&&r.list&&r.list.s?r.list.s.map(i=>Dt(i,s)):t}function Ce(a,e,t){const s=a[e];return s&&s.list&&s.list.shape?s.list.shape.map(r=>zt(r)):t}function Ve(a,e,t){const s=a[e];return s&&s.list&&s.list.b?s.list.b:t}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Yo{constructor(e,t,s){this.node=e,this.tensorMap=t,this.context=s,this.inputs=[],this.attrs={},this.inputs=e.inputNames.map(r=>this.getInput(r)),e.rawAttrs!=null&&(this.attrs=Object.keys(e.rawAttrs).reduce((r,i)=>(r[i]=this.getAttr(i),r),{}))}getInput(e){return A(e,this.tensorMap,this.context)}getAttr(e,t){const s=this.node.rawAttrs[e];if(s.tensor!=null)return A(e,this.tensorMap,this.context);if(s.i!=null||s.f!=null)return Ie(this.node.rawAttrs,e,t);if(s.s!=null)return Ae(this.node.rawAttrs,e,t);if(s.b!=null)return Ee(this.node.rawAttrs,e,t);if(s.shape!=null)return De(this.node.rawAttrs,e,t);if(s.type!=null)return ke(this.node.rawAttrs,e,t);if(s.list!=null){if(s.list.i!=null||s.list.f!=null)return ze(this.node.rawAttrs,e,t);if(s.list.s!=null)return Le(this.node.rawAttrs,e,t);if(s.list.shape!=null)return Ce(this.node.rawAttrs,e,t);if(s.list.b!=null)return Ve(this.node.rawAttrs,e,t);if(s.list.type!=null)return $e(this.node.rawAttrs,e,t)}return t}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const E=Object.freeze(Object.defineProperty({__proto__:null,OP_SCOPE_SUFFIX:Ca,abs:Va,acos:Pa,acosh:Fa,add:X,addN:wn,all:ja,any:xa,argMax:Ra,argMin:Ba,asin:Ha,asinh:Wa,atan:qa,atan2:Ua,atanh:Ga,avgPool:Ka,avgPool3d:Ja,basicLSTMCell:On,batchNorm:Xa,batchNorm2d:Za,batchNorm3d:Ya,batchNorm4d:Qa,batchToSpaceND:Ma,bincount:es,bitwiseAnd:_n,booleanMaskAsync:$i,broadcastArgs:En,broadcastTo:ts,buffer:nt,cast:Fe,ceil:as,clipByValue:ss,clone:Ot,complex:rs,concat:ge,concat1d:ns,concat2d:is,concat3d:os,concat4d:us,conv1d:ps,conv2d:ms,conv2dTranspose:ls,conv3d:cs,conv3dTranspose:ds,cos:hs,cosh:ys,cosineWindow:fs,cumprod:gs,cumsum:Ns,denseBincount:bs,depthToSpace:Ts,depthwiseConv2d:ht,diag:kn,dilation2d:ws,div:ct,divNoNan:Ss,dot:Os,dropout:vs,einsum:_s,elu:As,enclosingPowerOfTwo:Es,ensureShape:Dn,equal:Is,erf:ks,euclideanNorm:$s,exp:Ds,expandDims:zs,expm1:Ls,eye:Cs,fft:Vs,fill:Ps,floor:Fs,floorDiv:js,fused:qi,gather:lt,gatherND:xi,greater:xs,greaterEqual:Rs,ifft:Bs,imag:Hs,image:Ws,inTopKAsync:Bi,irfft:qs,isFinite:Us,isInf:Gs,isNaN:Ks,leakyRelu:Js,less:Xs,lessEqual:Zs,linalg:Ys,linspace:zn,localResponseNormalization:Qs,log:Ms,log1p:er,logSigmoid:tr,logSoftmax:ar,logSumExp:sr,logicalAnd:rr,logicalNot:nr,logicalOr:ir,logicalXor:or,losses:ur,lowerBound:Cn,matMul:q,max:pr,maxPool:mr,maxPool3d:lr,maxPoolWithArgmax:Pn,maximum:cr,mean:dr,meshgrid:Fn,min:hr,minimum:yr,mirrorPad:fr,mod:gr,moments:Nr,movingAverage:zi,mul:te,multiRNNCell:xn,multinomial:Bn,neg:br,norm:Tr,notEqual:wr,oneHot:Sr,ones:ee,onesLike:Or,op:T,outerProduct:Wn,pad:re,pad1d:Un,pad2d:Kn,pad3d:Xn,pad4d:Yn,pool:vr,pow:dt,prelu:_r,print:Ar,prod:Er,raggedGather:Mn,raggedRange:ti,raggedTensorToTensor:si,rand:ni,randomGamma:oi,randomNormal:it,randomStandardNormal:pi,randomUniform:ot,randomUniformInt:li,range:Ir,real:kr,reciprocal:$r,relu:Dr,relu6:zr,reshape:O,reverse:ne,reverse1d:di,reverse2d:yi,reverse3d:gi,reverse4d:bi,rfft:Lr,round:Cr,rsqrt:Vr,scalar:B,scatterND:Ci,searchSorted:je,selu:Pr,separableConv2d:Fr,setdiff1dAsync:wi,sigmoid:pe,sign:jr,signal:xr,sin:Rr,sinh:Br,slice:U,slice1d:Hr,slice2d:Wr,slice3d:qr,slice4d:Ur,softmax:Gr,softplus:Kr,spaceToBatchND:Jr,sparse:Xr,sparseToDense:Fi,spectral:Zr,split:Yr,sqrt:Qr,square:Mr,squaredDifference:en,squeeze:mt,stack:se,step:tn,stridedSlice:an,string:sn,sub:me,sum:rn,tan:nn,tanh:_e,tensor:Y,tensor1d:on,tensor2d:un,tensor3d:vt,tensor4d:Si,tensor5d:Oi,tensor6d:vi,tensorScatterUpdate:Ai,tile:pn,topk:mn,transpose:ln,truncatedNormal:cn,unique:dn,unsortedSegmentSum:hn,unstack:ie,upperBound:Ei,variable:yn,where:fn,whereAsync:_t,zeros:gn,zerosLike:Nn},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Qo=(a,e,t,s=E)=>{switch(a.op){case"BiasAdd":case"AddV2":case"Add":return[s.add(n("a",a,e,t),n("b",a,e,t))];case"AddN":return[s.addN(n("tensors",a,e,t))];case"FloorMod":case"Mod":return[s.mod(n("a",a,e,t),n("b",a,e,t))];case"Mul":return[s.mul(n("a",a,e,t),n("b",a,e,t))];case"RealDiv":case"Div":return[s.div(n("a",a,e,t),n("b",a,e,t))];case"DivNoNan":return[s.divNoNan(n("a",a,e,t),n("b",a,e,t))];case"FloorDiv":return[s.floorDiv(n("a",a,e,t),n("b",a,e,t))];case"Sub":return[s.sub(n("a",a,e,t),n("b",a,e,t))];case"Minimum":return[s.minimum(n("a",a,e,t),n("b",a,e,t))];case"Maximum":return[s.maximum(n("a",a,e,t),n("b",a,e,t))];case"Pow":return[s.pow(n("a",a,e,t),n("b",a,e,t))];case"SquaredDifference":return[s.squaredDifference(n("a",a,e,t),n("b",a,e,t))];default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Mo=(a,e,t,s=E)=>{switch(a.op){case"Abs":case"ComplexAbs":return[s.abs(n("x",a,e,t))];case"Acos":return[s.acos(n("x",a,e,t))];case"Acosh":return[s.acosh(n("x",a,e,t))];case"Asin":return[s.asin(n("x",a,e,t))];case"Asinh":return[s.asinh(n("x",a,e,t))];case"Atan":return[s.atan(n("x",a,e,t))];case"Atan2":return[s.atan2(n("x",a,e,t),n("y",a,e,t))];case"Atanh":return[s.atanh(n("x",a,e,t))];case"Ceil":return[s.ceil(n("x",a,e,t))];case"Complex":return[s.complex(n("real",a,e,t),n("imag",a,e,t))];case"Cos":return[s.cos(n("x",a,e,t))];case"Cosh":return[s.cosh(n("x",a,e,t))];case"Elu":return[s.elu(n("x",a,e,t))];case"Erf":return[s.erf(n("x",a,e,t))];case"Exp":return[s.exp(n("x",a,e,t))];case"Expm1":return[s.expm1(n("x",a,e,t))];case"Floor":return[s.floor(n("x",a,e,t))];case"Log":return[s.log(n("x",a,e,t))];case"Log1p":return[s.log1p(n("x",a,e,t))];case"Imag":return[s.imag(n("x",a,e,t))];case"Neg":return[s.neg(n("x",a,e,t))];case"Reciprocal":return[s.reciprocal(n("x",a,e,t))];case"Real":return[s.real(n("x",a,e,t))];case"Relu":return[s.relu(n("x",a,e,t))];case"Round":return[s.round(n("x",a,e,t))];case"Selu":return[s.selu(n("x",a,e,t))];case"Sigmoid":return[s.sigmoid(n("x",a,e,t))];case"Sin":return[s.sin(n("x",a,e,t))];case"Sign":return[s.sign(n("x",a,e,t))];case"Sinh":return[s.sinh(n("x",a,e,t))];case"Softplus":return[s.softplus(n("x",a,e,t))];case"Sqrt":return[s.sqrt(n("x",a,e,t))];case"Square":return[s.square(n("x",a,e,t))];case"Tanh":return[s.tanh(n("x",a,e,t))];case"Tan":return[s.tan(n("x",a,e,t))];case"ClipByValue":return[s.clipByValue(n("x",a,e,t),n("clipValueMin",a,e,t),n("clipValueMax",a,e,t))];case"Relu6":return[s.relu6(n("x",a,e,t))];case"Rsqrt":return[s.rsqrt(A(a.inputNames[0],e,t))];case"LeakyRelu":return[s.leakyRelu(n("x",a,e,t),n("alpha",a,e,t))];case"Prelu":return[s.prelu(n("x",a,e,t),n("alpha",a,e,t))];case"IsNan":return[s.isNaN(A(a.inputNames[0],e,t))];case"IsInf":return[s.isInf(A(a.inputNames[0],e,t))];case"IsFinite":return[s.isFinite(A(a.inputNames[0],e,t))];default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function C(a,e,t=""){if(!(typeof a=="number"||typeof e=="number")){S(a.length===e.length,()=>t+` Shapes ${a} and ${e} must match`);for(let s=0;s<a.length;s++){const r=a[s],i=e[s];S(r<0||i<0||r===i,()=>t+` Shapes ${a} and ${e} must match`)}}}function et(a){return!(typeof a=="number"||a.some(e=>e<0))}function M(a,e,t){let s=Pe(a,t);const r=!et(s);if(r&&e.length===0)throw new Error(`Tried to calculate elements of an empty list with non-fully-defined elementShape: ${s}`);if(r&&e.forEach(i=>{s=Pe(i.shape,s)}),!et(s))throw new Error(`Non-fully-defined elementShape: ${s}`);return s}function Pe(a,e){if(typeof a=="number")return e;if(typeof e=="number")return a;if(a.length!==e.length)throw new Error(`Incompatible ranks during merge: ${a} vs. ${e}`);const t=[];for(let s=0;s<a.length;++s){const r=a[s],i=e[s];if(r>=0&&i>=0&&r!==i)throw new Error(`Incompatible shape during merge: ${a} vs. ${e}`);t[s]=r>=0?r:i}return t}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class eu{constructor(e,t,s,r,i,u,o){this.name=e,this.dtype=t,this.maxSize=s,this.elementShape=r,this.identicalElementShapes=i,this.dynamicSize=u,this.clearAfterRead=o,this.tensors=[],this.closed_=!1,this.idTensor=B(0),R(this.idTensor)}get id(){return this.idTensor.id}get closed(){return this.closed_}clearAndClose(e){this.tensors.forEach(t=>{(e==null||!e.has(t.tensor.id))&&t.tensor.dispose()}),this.tensors=[],this.closed_=!0,this.idTensor.dispose()}size(){return this.tensors.length}read(e){if(this.closed_)throw new Error(`TensorArray ${this.name} has already been closed.`);if(e<0||e>=this.size())throw new Error(`Tried to read from index ${e}, but array size is: ${this.size()}`);const t=this.tensors[e];if(t.cleared)throw new Error(`TensorArray ${this.name}: Could not read index ${e} twice because it was cleared after a previous read (perhaps try setting clear_after_read = false?).`);return this.clearAfterRead&&(t.cleared=!0),t.read=!0,t.tensor}readMany(e){return e.map(t=>this.read(t))}write(e,t){if(this.closed_)throw new Error(`TensorArray ${this.name} has already been closed.`);if(e<0||!this.dynamicSize&&e>=this.maxSize)throw new Error(`Tried to write to index ${e}, but array is not resizeable and size is: ${this.maxSize}`);const s=this.tensors[e]||{};if(t.dtype!==this.dtype)throw new Error(`TensorArray ${this.name}: Could not write to TensorArray index ${e},
          because the value dtype is ${t.dtype}, but TensorArray dtype is ${this.dtype}.`);if(this.size()===0&&(this.elementShape==null||this.elementShape.length===0)&&(this.elementShape=t.shape),C(this.elementShape,t.shape,`TensorArray ${this.name}: Could not write to TensorArray index ${e}.`),s.read)throw new Error(`TensorArray ${this.name}: Could not write to TensorArray index ${e}, because it has already been read.`);if(s.written)throw new Error(`TensorArray ${this.name}: Could not write to TensorArray index ${e}, because it has already been written.`);s.tensor=t,R(t),s.written=!0,this.tensors[e]=s}writeMany(e,t){if(e.length!==t.length)throw new Error(`TensorArray ${this.name}: could not write multiple tensors,because the index size: ${e.length} is not the same as tensors size: ${t.length}.`);e.forEach((s,r)=>this.write(s,t[r]))}gather(e,t){if(t&&t!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but gather requested dtype ${t}`);if(e)e=e.slice(0,this.size());else{e=[];for(let r=0;r<this.size();r++)e.push(r)}if(e.length===0)return Y([],[0].concat(this.elementShape));const s=this.readMany(e);return C(this.elementShape,s[0].shape,"TensorArray shape mismatch: "),se(s,0)}concat(e){if(e&&e!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but concat requested dtype ${e}`);if(this.size()===0)return Y([],[0].concat(this.elementShape));const t=[];for(let r=0;r<this.size();r++)t.push(r);const s=this.readMany(t);return C(this.elementShape,s[0].shape,`TensorArray shape mismatch: tensor array shape (${this.elementShape}) vs first tensor shape (${s[0].shape})`),ge(s,0)}scatter(e,t){if(t.dtype!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but tensor has dtype ${t.dtype}`);if(e.length!==t.shape[0])throw new Error(`Expected len(indices) == tensor.shape[0], but saw: ${e.length} vs. ${t.shape[0]}`);const s=Math.max(...e);if(!this.dynamicSize&&s>=this.maxSize)throw new Error(`Max index must be < array size (${s}  vs. ${this.maxSize})`);this.writeMany(e,ie(t,0))}split(e,t){if(t.dtype!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but tensor has dtype ${t.dtype}`);let s=0;const r=e.map(p=>(s+=p,s));if(s!==t.shape[0])throw new Error(`Expected sum of lengths to be equal to
          tensor.shape[0], but sum of lengths is
        ${s}, and tensor's shape is: ${t.shape}`);if(!this.dynamicSize&&e.length!==this.maxSize)throw new Error(`TensorArray's size is not equal to the size of lengths (${this.maxSize} vs. ${e.length}), and the TensorArray is not marked as dynamically resizeable`);const i=s===0?0:t.size/s,u=[];P(()=>{t=O(t,[1,s,i]);for(let p=0;p<e.length;++p){const l=[0,p===0?0:r[p-1],0],c=[1,e[p],i];u[p]=O(U(t,l,c),this.elementShape)}return u});const o=[];for(let p=0;p<e.length;p++)o[p]=p;this.writeMany(o,u)}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class K{get id(){return this.idTensor.id}constructor(e,t,s,r=-1){this.tensors=e,this.elementShape=t,this.elementDtype=s,e?.forEach(i=>{if(s!==i.dtype)throw new Error(`Invalid data types; op elements ${s}, but list elements ${i.dtype}`);C(t,i.shape,"TensorList shape mismatch: "),R(i)}),this.idTensor=B(0),this.maxNumElements=r,R(this.idTensor)}copy(){return new K([...this.tensors],this.elementShape,this.elementDtype)}clearAndClose(e){this.tensors.forEach(t=>{(e==null||!e.has(t.id))&&t.dispose()}),this.tensors.length=0,this.idTensor.dispose()}size(){return this.tensors.length}stack(e,t,s=-1){if(t!==this.elementDtype)throw new Error(`Invalid data types; op elements ${t}, but list elements ${this.elementDtype}`);if(s!==-1&&this.tensors.length!==s)throw new Error(`Operation expected a list with ${s} elements but got a list with ${this.tensors.length} elements.`);C(e,this.elementShape,"TensorList shape mismatch: ");const r=M(this.elementShape,this.tensors,e);return P(()=>{const i=this.tensors.map(u=>O(u,r));return se(i,0)})}popBack(e,t){if(t!==this.elementDtype)throw new Error(`Invalid data types; op elements ${t}, but list elements ${this.elementDtype}`);if(this.size()===0)throw new Error("Trying to pop from an empty list.");const s=M(this.elementShape,this.tensors,e),r=this.tensors.pop();return r.kept=!1,C(r.shape,e,"TensorList shape mismatch: "),O(r,s)}pushBack(e){if(e.dtype!==this.elementDtype)throw new Error(`Invalid data types; op elements ${e.dtype}, but list elements ${this.elementDtype}`);if(C(e.shape,this.elementShape,"TensorList shape mismatch: "),this.maxNumElements===this.size())throw new Error("Trying to push element into a full list.");R(e),this.tensors.push(e)}resize(e){if(e<0)throw new Error(`TensorListResize expects size to be non-negative. Got: ${e}`);if(this.maxNumElements!==-1&&e>this.maxNumElements)throw new Error(`TensorListResize input size ${e} is greater maxNumElement ${this.maxNumElements}.`);const t=new K([],this.elementShape,this.elementDtype,this.maxNumElements);t.tensors.length=e;for(let s=0;s<Math.min(this.tensors.length,e);++s)t.tensors[s]=this.tensors[s];return t}getItem(e,t,s){if(s!==this.elementDtype)throw new Error(`Invalid data types; op elements ${s}, but list elements ${this.elementDtype}`);if(e<0||e>this.tensors.length)throw new Error(`Trying to access element ${e} in a list with ${this.tensors.length} elements.`);if(this.tensors[e]==null)throw new Error(`element at index ${e} is null.`);C(this.tensors[e].shape,t,"TensorList shape mismatch: ");const r=M(this.elementShape,this.tensors,t);return O(this.tensors[e],r)}setItem(e,t){if(t.dtype!==this.elementDtype)throw new Error(`Invalid data types; op elements ${t.dtype}, but list elements ${this.elementDtype}`);if(e<0||this.maxNumElements!==-1&&e>=this.maxNumElements)throw new Error(`Trying to set element ${e} in a list with max ${this.maxNumElements} elements.`);C(this.elementShape,t.shape,"TensorList shape mismatch: "),R(t),this.tensors[e]!=null&&(this.tensors[e].kept=!1),this.tensors[e]=t}gather(e,t,s){if(t!==this.elementDtype)throw new Error(`Invalid data types; op elements ${t}, but list elements ${this.elementDtype}`);C(this.elementShape,s,"TensorList shape mismatch: "),e=e.slice(0,this.size());const r=M(this.elementShape,this.tensors,s);return e.length===0?Y([],[0].concat(r)):P(()=>{const i=e.map(u=>O(this.tensors[u],r));return se(i,0)})}concat(e,t){if(e&&e!==this.elementDtype)throw new Error(`TensorList dtype is ${this.elementDtype} but concat requested dtype ${e}`);C(this.elementShape,t,"TensorList shape mismatch: ");const s=M(this.elementShape,this.tensors,t);return this.size()===0?Y([],[0].concat(s)):P(()=>{const r=this.tensors.map(i=>O(i,s));return ge(r,0)})}}function tu(a,e,t){const s=a.dtype;if(a.shape.length<1)throw new Error(`Tensor must be at least a vector, but saw shape: ${a.shape}`);if(a.dtype!==t)throw new Error(`Invalid data types; op elements ${a.dtype}, but list elements ${t}`);const r=a.shape.slice(1);C(r,e,"TensorList shape mismatch: ");const i=ie(a);return new K(i,e,s)}function au(a,e,t,s){return new K([],a,e,s)}function su(a,e,t,s){if(e.length!==a.shape[0])throw new Error(`Expected len(indices) == tensor.shape[0], but saw: ${e.length} vs. ${a.shape[0]}`);const r=Math.max(...e);if(s!=null&&s!==-1&&r>=s)throw new Error(`Max index must be < array size (${r}  vs. ${s})`);const i=new K([],t,a.dtype,s),u=ie(a,0);return e.forEach((o,p)=>{i.setItem(o,u[p])}),i}function ru(a,e,t){let s=0;const r=e.map(l=>(s+=l,s));if(s!==a.shape[0])throw new Error(`Expected sum of lengths to be equal to
          tensor.shape[0], but sum of lengths is
        ${s}, and tensor's shape is: ${a.shape}`);const i=a.shape.slice(1),u=Pe(i,t),o=s===0?0:a.size/s,p=P(()=>{const l=[];a=O(a,[1,s,o]);for(let c=0;c<e.length;++c){const h=[0,c===0?0:r[c-1],0],N=[1,e[c],o];l[c]=O(U(a,h,N),u)}return a.dispose(),l}),m=new K([],t,a.dtype,e.length);for(let l=0;l<p.length;l++)m.setItem(l,p[l]);return m}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const nu=async(a,e,t)=>{switch(a.op){case"If":case"StatelessIf":{const s=n("thenBranch",a,e,t),r=n("elseBranch",a,e,t),i=n("cond",a,e,t),u=n("args",a,e,t);return(await i.data())[0]?t.functionMap[s].executeFunctionAsync(u,t.tensorArrayMap,t.tensorListMap):t.functionMap[r].executeFunctionAsync(u,t.tensorArrayMap,t.tensorListMap)}case"While":case"StatelessWhile":{const s=n("body",a,e,t),r=n("cond",a,e,t),i=n("args",a,e,t),u=await t.functionMap[r].executeFunctionAsync(i,t.tensorArrayMap,t.tensorListMap),o=i.map(l=>l.id);let p=await u[0].data();u.forEach(l=>{!l.kept&&o.indexOf(l.id)===-1&&l.dispose()});let m=i;for(;p[0];){const l=m;m=await t.functionMap[s].executeFunctionAsync(m,t.tensorArrayMap,t.tensorListMap);const c=m.map(h=>h.id);l.forEach(h=>{!h.kept&&o.indexOf(h.id)===-1&&c.indexOf(h.id)===-1&&h.dispose()});const d=await t.functionMap[r].executeFunctionAsync(m,t.tensorArrayMap,t.tensorListMap);p=await d[0].data(),d.forEach(h=>{!h.kept&&o.indexOf(h.id)===-1&&c.indexOf(h.id)===-1&&h.dispose()})}return m}case"LoopCond":{const s=n("pred",a,e,t);return[x(s)]}case"Switch":{const s=n("pred",a,e,t);let r=n("data",a,e,t);return r.kept||(r=x(r)),(await s.data())[0]?[void 0,r]:[r,void 0]}case"Merge":{const s=a.inputNames.find(r=>A(r,e,t)!==void 0);if(s){const r=A(s,e,t);return[x(r)]}return}case"Enter":{const s=n("frameName",a,e,t),r=n("tensor",a,e,t);return t.enterFrame(s),[x(r)]}case"Exit":{const s=n("tensor",a,e,t);return t.exitFrame(),[x(s)]}case"NextIteration":{const s=n("tensor",a,e,t);return t.nextIteration(),[x(s)]}case"TensorArrayV3":{const s=n("size",a,e,t),r=n("dtype",a,e,t),i=n("elementShape",a,e,t),u=n("dynamicSize",a,e,t),o=n("clearAfterRead",a,e,t),p=n("identicalElementShapes",a,e,t),m=n("name",a,e,t),l=new eu(m,r,s,i,p,u,o);return t.addTensorArray(l),[l.idTensor,B(1)]}case"TensorArrayWriteV3":{const s=n("tensorArrayId",a,e,t),r=n("index",a,e,t),i=n("tensor",a,e,t),u=t.getTensorArray(s.id);return u.write(r,i),[u.idTensor]}case"TensorArrayReadV3":{const s=n("tensorArrayId",a,e,t),r=n("index",a,e,t);return[t.getTensorArray(s.id).read(r)]}case"TensorArrayGatherV3":{const s=n("tensorArrayId",a,e,t),r=n("indices",a,e,t),i=n("dtype",a,e,t);return[t.getTensorArray(s.id).gather(r,i)]}case"TensorArrayScatterV3":{const s=n("tensorArrayId",a,e,t),r=n("indices",a,e,t),i=n("tensor",a,e,t),u=t.getTensorArray(s.id);return u.scatter(r,i),[u.idTensor]}case"TensorArrayConcatV3":{const s=n("tensorArrayId",a,e,t),r=t.getTensorArray(s.id),i=n("dtype",a,e,t);return[r.concat(i)]}case"TensorArraySplitV3":{const s=n("tensorArrayId",a,e,t),r=n("tensor",a,e,t),i=n("lengths",a,e,t),u=t.getTensorArray(s.id);return u.split(i,r),[u.idTensor]}case"TensorArraySizeV3":{const s=n("tensorArrayId",a,e,t),r=t.getTensorArray(s.id);return[B(r.size(),"int32")]}case"TensorArrayCloseV3":{const s=n("tensorArrayId",a,e,t),r=t.getTensorArray(s.id);return r.clearAndClose(),[r.idTensor]}case"TensorListSetItem":{const s=n("tensorListId",a,e,t),r=n("index",a,e,t),i=n("tensor",a,e,t),u=t.getTensorList(s.id);return u.setItem(r,i),[u.idTensor]}case"TensorListGetItem":{const s=n("tensorListId",a,e,t),r=n("index",a,e,t),i=n("elementShape",a,e,t),u=n("elementDType",a,e,t);return[t.getTensorList(s.id).getItem(r,i,u)]}case"TensorListScatterV2":case"TensorListScatter":{const s=n("indices",a,e,t),r=n("tensor",a,e,t),i=n("elementShape",a,e,t),u=n("numElements",a,e,t),o=su(r,s,i,u);return t.addTensorList(o),[o.idTensor]}case"TensorListReserve":case"EmptyTensorList":{const s=n("elementShape",a,e,t),r=n("elementDType",a,e,t);let i;a.op==="TensorListReserve"?i="numElements":i="maxNumElements";const u=n(i,a,e,t),o=a.op==="TensorListReserve"?-1:u,p=au(s,r,u,o);return t.addTensorList(p),[p.idTensor]}case"TensorListGather":{const s=n("tensorListId",a,e,t),r=n("indices",a,e,t),i=n("elementShape",a,e,t),u=n("elementDType",a,e,t);return[t.getTensorList(s.id).gather(r,u,i)]}case"TensorListStack":{const s=n("tensorListId",a,e,t),r=n("elementShape",a,e,t),i=n("elementDType",a,e,t),u=n("numElements",a,e,t);return[t.getTensorList(s.id).stack(r,i,u)]}case"TensorListFromTensor":{const s=n("tensor",a,e,t),r=n("elementShape",a,e,t),i=n("elementDType",a,e,t),u=tu(s,r,i);return t.addTensorList(u),[u.idTensor]}case"TensorListConcat":case"TensorListConcatV2":{const s=n("tensorListId",a,e,t),r=t.getTensorList(s.id),i=n("dtype",a,e,t),u=n("elementShape",a,e,t);return[r.concat(i,u)]}case"TensorListPushBack":{const s=n("tensorListId",a,e,t),r=n("tensor",a,e,t),i=t.getTensorList(s.id);return i.pushBack(r),[i.idTensor]}case"TensorListPopBack":{const s=n("tensorListId",a,e,t),r=n("elementShape",a,e,t),i=n("elementDType",a,e,t);return[t.getTensorList(s.id).popBack(r,i)]}case"TensorListSplit":{const s=n("tensor",a,e,t),r=n("elementShape",a,e,t),i=n("lengths",a,e,t),u=ru(s,i,r);return t.addTensorList(u),[u.idTensor]}case"TensorListLength":{const s=n("tensorListId",a,e,t),r=t.getTensorList(s.id);return[B(r.size(),"int32")]}case"TensorListResize":{const s=n("tensorListId",a,e,t),r=n("size",a,e,t),u=t.getTensorList(s.id).resize(r);return t.addTensorList(u),[u.idTensor]}default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function tt(a,e,t){const[s,r]=n("fusedOps",a,e,t),i=s==="biasadd",u=!i,o=r==="prelu",p=s==="fusedbatchnorm",m=n("numArgs",a,e,t);if(i){if(o&&m!==2)throw new Error("FusedConv2d and DepthwiseConv2d with BiasAdd and Prelu must have two extra arguments: bias and alpha.");if(!o&&i&&m!==1)throw new Error("FusedConv2d and DepthwiseConv2d with BiasAdd must have one extra argument: bias.")}if(p)throw new Error("FusedConv2d and DepthwiseConv2d with FusedBatchNorm is not supported");const l=n("strides",a,e,t),c=le(a,e,t),d=n("dataFormat",a,e,t).toUpperCase(),h=n("dilations",a,e,t);let[N,f]=n("args",a,e,t);u&&(f=N,N=void 0);const y=n("leakyreluAlpha",a,e,t);return{stride:l,pad:c,dataFormat:d,dilations:h,biasArg:N,preluArg:f,activationFunc:r,leakyreluAlpha:y}}const iu=(a,e,t,s=E)=>{switch(a.op){case"Conv1D":{const r=n("stride",a,e,t),i=n("pad",a,e,t),u=n("dataFormat",a,e,t).toUpperCase(),o=n("dilation",a,e,t);return[s.conv1d(n("x",a,e,t),n("filter",a,e,t),r,i,u,o)]}case"Conv2D":{const r=n("strides",a,e,t),i=le(a,e,t),u=n("dataFormat",a,e,t).toUpperCase(),o=n("dilations",a,e,t);return[s.conv2d(n("x",a,e,t),n("filter",a,e,t),[r[1],r[2]],i,u,[o[1],o[2]])]}case"_FusedConv2D":{const{stride:r,pad:i,dataFormat:u,dilations:o,biasArg:p,preluArg:m,activationFunc:l,leakyreluAlpha:c}=tt(a,e,t);return[s.fused.conv2d({x:n("x",a,e,t),filter:n("filter",a,e,t),strides:[r[1],r[2]],pad:i,dataFormat:u,dilations:[o[1],o[2]],bias:p,activation:l,preluActivationWeights:m,leakyreluAlpha:c})]}case"FusedDepthwiseConv2dNative":{const{stride:r,pad:i,dataFormat:u,dilations:o,biasArg:p,preluArg:m,activationFunc:l,leakyreluAlpha:c}=tt(a,e,t);return[s.fused.depthwiseConv2d({x:n("x",a,e,t),filter:n("filter",a,e,t),strides:[r[1],r[2]],pad:i,dataFormat:u,dilations:[o[1],o[2]],bias:p,activation:l,preluActivationWeights:m,leakyreluAlpha:c})]}case"Conv2DBackpropInput":case"Conv2dTranspose":{const r=n("outputShape",a,e,t),i=n("strides",a,e,t),u=le(a,e,t);return[s.conv2dTranspose(n("x",a,e,t),n("filter",a,e,t),r,[i[1],i[2]],u)]}case"DepthwiseConv2dNative":case"DepthwiseConv2d":{const r=n("strides",a,e,t),i=le(a,e,t),u=n("dilations",a,e,t),o=n("dataFormat",a,e,t).toUpperCase();return[s.depthwiseConv2d(n("input",a,e,t),n("filter",a,e,t),[r[1],r[2]],i,o,[u[1],u[2]])]}case"Conv3D":{const r=n("strides",a,e,t),i=n("pad",a,e,t),u=n("dataFormat",a,e,t).toUpperCase(),o=n("dilations",a,e,t);return[s.conv3d(n("x",a,e,t),n("filter",a,e,t),[r[1],r[2],r[3]],i,u,[o[1],o[2],o[3]])]}case"AvgPool":{const r=n("strides",a,e,t),i=n("pad",a,e,t),u=n("kernelSize",a,e,t);return[s.avgPool(n("x",a,e,t),[u[1],u[2]],[r[1],r[2]],i)]}case"MaxPool":{const r=n("strides",a,e,t),i=n("pad",a,e,t),u=n("kernelSize",a,e,t);return[s.maxPool(n("x",a,e,t),[u[1],u[2]],[r[1],r[2]],i)]}case"MaxPoolWithArgmax":{const r=n("strides",a,e,t),i=n("pad",a,e,t),u=n("kernelSize",a,e,t),o=n("includeBatchInIndex",a,e,t),{result:p,indexes:m}=s.maxPoolWithArgmax(n("x",a,e,t),[u[1],u[2]],[r[1],r[2]],i,o);return[p,m]}case"AvgPool3D":{const r=n("strides",a,e,t),i=n("pad",a,e,t),u=n("kernelSize",a,e,t);return[s.avgPool3d(n("x",a,e,t),[u[1],u[2],u[3]],[r[1],r[2],r[3]],i)]}case"MaxPool3D":{const r=n("strides",a,e,t),i=n("pad",a,e,t),u=n("kernelSize",a,e,t);return[s.maxPool3d(n("x",a,e,t),[u[1],u[2],u[3]],[r[1],r[2],r[3]],i)]}case"Dilation2D":{const r=n("strides",a,e,t),i=n("pad",a,e,t),u=n("dilations",a,e,t),o=r[1],p=r[2],m=u[1],l=u[2];return[s.dilation2d(n("x",a,e,t),n("filter",a,e,t),[o,p],i,[m,l],"NHWC")]}default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ou=(a,e,t,s=E)=>{switch(a.op){case"Fill":{const r=n("shape",a,e,t),i=n("dtype",a,e,t),u=n("value",a,e,t);return[s.fill(r,u,i)]}case"LinSpace":{const r=n("start",a,e,t),i=n("stop",a,e,t),u=n("num",a,e,t);return[s.linspace(r,i,u)]}case"Multinomial":{const r=n("logits",a,e,t),i=n("numSamples",a,e,t),u=n("seed",a,e,t);return[s.multinomial(r,i,u)]}case"OneHot":{const r=n("indices",a,e,t),i=n("depth",a,e,t),u=n("onValue",a,e,t),o=n("offValue",a,e,t),p=n("dtype",a,e,t);return[s.oneHot(r,i,u,o,p)]}case"Ones":return[s.ones(n("shape",a,e,t),n("dtype",a,e,t))];case"OnesLike":return[s.onesLike(n("x",a,e,t))];case"RandomStandardNormal":return[s.randomStandardNormal(n("shape",a,e,t),n("dtype",a,e,t),n("seed",a,e,t))];case"RandomUniform":return[s.randomUniform(n("shape",a,e,t),n("minval",a,e,t),n("maxval",a,e,t),n("dtype",a,e,t))];case"RandomUniformInt":return[s.randomUniformInt(n("shape",a,e,t),n("minval",a,e,t),n("maxval",a,e,t),n("seed",a,e,t))];case"Range":{const r=n("start",a,e,t),i=n("stop",a,e,t),u=n("step",a,e,t);return[s.range(r,i,u,n("dtype",a,e,t))]}case"TruncatedNormal":{const r=n("shape",a,e,t),i=n("mean",a,e,t),u=n("stdDev",a,e,t),o=n("seed",a,e,t);return[s.truncatedNormal(r,i,u,n("dtype",a,e,t),o)]}case"Zeros":return[s.zeros(n("shape",a,e,t),n("dtype",a,e,t))];case"ZerosLike":return[s.zerosLike(n("x",a,e,t))];default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ve(a,e,t){const s=n("boxes",a,e,t),r=n("scores",a,e,t),i=n("maxOutputSize",a,e,t),u=n("iouThreshold",a,e,t),o=n("scoreThreshold",a,e,t),p=n("softNmsSigma",a,e,t);return{boxes:s,scores:r,maxOutputSize:i,iouThreshold:u,scoreThreshold:o,softNmsSigma:p}}const uu=async(a,e,t,s,r=E)=>{switch(a.op){case"NonMaxSuppressionV5":{const{boxes:i,scores:u,maxOutputSize:o,iouThreshold:p,scoreThreshold:m,softNmsSigma:l}=ve(a,e,t),c=await r.image.nonMaxSuppressionWithScoreAsync(i,u,o,p,m,l);return[c.selectedIndices,c.selectedScores]}case"NonMaxSuppressionV4":{const{boxes:i,scores:u,maxOutputSize:o,iouThreshold:p,scoreThreshold:m}=ve(a,e,t),l=n("padToMaxOutputSize",a,e,t),c=await r.image.nonMaxSuppressionPaddedAsync(i,u,o,p,m,l);return[c.selectedIndices,c.validOutputs]}case"NonMaxSuppressionV3":case"NonMaxSuppressionV2":{const{boxes:i,scores:u,maxOutputSize:o,iouThreshold:p,scoreThreshold:m}=ve(a,e,t);return[await r.image.nonMaxSuppressionAsync(i,u,o,p,m)]}case"Where":{const i=r.cast(n("condition",a,e,t),"bool"),u=[await r.whereAsync(i)];return i.dispose(),u}case"ListDiff":return r.setdiff1dAsync(n("x",a,e,t),n("y",a,e,t));default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const pu=(a,e,t,s=E)=>{switch(a.op){case"LowerBound":{const r=n("sortedSequence",a,e,t),i=n("values",a,e,t);return[s.lowerBound(r,i)]}case"TopKV2":{const r=n("x",a,e,t),i=n("k",a,e,t),u=n("sorted",a,e,t),o=s.topk(r,i,u);return[o.values,o.indices]}case"UpperBound":{const r=n("sortedSequence",a,e,t),i=n("values",a,e,t);return[s.upperBound(r,i)]}case"Unique":{const r=n("x",a,e,t),i=s.unique(r);return[i.values,i.indices]}case"UniqueV2":{const r=n("x",a,e,t),i=n("axis",a,e,t),u=s.unique(r,i);return[u.values,u.indices]}default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const mu=(a,e,t,s=E)=>{switch(a.op){case"Const":return e[a.name];case"PlaceholderWithDefault":const r=n("default",a,e,t);return[A(a.name,e,t)||r];case"Placeholder":return[A(a.name,e,t)];case"Identity":case"StopGradient":case"FakeQuantWithMinMaxVars":{const l=n("x",a,e,t);return[x(l)]}case"IdentityN":return n("x",a,e,t).map(l=>x(l));case"Snapshot":const i=n("x",a,e,t);return[x(i)];case"Shape":return[s.tensor1d(n("x",a,e,t).shape,"int32")];case"ShapeN":return n("x",a,e,t).map(l=>s.tensor1d(l.shape));case"Size":return[s.scalar(n("x",a,e,t).size,"int32")];case"Rank":return[s.scalar(n("x",a,e,t).rank,"int32")];case"NoOp":return[s.scalar(1)];case"Print":const u=n("x",a,e,t),o=n("data",a,e,t),p=n("message",a,e,t),m=n("summarize",a,e,t);console.warn("The graph has a tf.print() operation,usually used for debugging, which slows down performance."),console.log(p);for(let l=0;l<o.length;l++)console.log(Array.prototype.slice.call(o[l].dataSync()).slice(0,m));return[u];default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class lu{get id(){return this.handle.id}constructor(e,t){this.keyDType=e,this.valueDType=t,this.handle=B(0),this.tensorMap=new Map,R(this.handle)}clearAndClose(){this.tensorMap.forEach(e=>e.dispose()),this.tensorMap.clear(),this.handle.dispose()}size(){return this.tensorMap.size}tensorSize(){return B(this.size(),"int32")}async import(e,t){this.checkKeyAndValueTensor(e,t);const s=await e.data();return this.tensorMap.forEach(r=>r.dispose()),this.tensorMap.clear(),P(()=>{const r=ie(t),i=s.length,u=r.length;S(i===u,()=>`The number of elements doesn't match, keys has ${i} elements, the values has ${u} elements.`);for(let o=0;o<i;o++){const p=s[o],m=r[o];R(m),this.tensorMap.set(p,m)}return this.handle})}async find(e,t){this.checkKeyAndValueTensor(e,t);const s=await e.data();return P(()=>{const r=[];for(let i=0;i<s.length;i++){const u=s[i],o=this.findWithDefault(u,t);r.push(o)}return se(r)})}findWithDefault(e,t){const s=this.tensorMap.get(e);return s??t}checkKeyAndValueTensor(e,t){if(e.dtype!==this.keyDType)throw new Error(`Expect key dtype ${this.keyDType}, but got ${e.dtype}`);if(t.dtype!==this.valueDType)throw new Error(`Expect value dtype ${this.valueDType}, but got ${t.dtype}`)}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const cu=async(a,e,t,s)=>{switch(a.op){case"HashTable":case"HashTableV2":{const r=s.getHashTableHandleByName(a.name);if(r!=null)return[r];{const i=n("keyDType",a,e,t),u=n("valueDType",a,e,t),o=new lu(i,u);return s.addHashTable(a.name,o),[o.handle]}}case"InitializeTable":case"InitializeTableV2":case"LookupTableImport":case"LookupTableImportV2":{const r=n("tableHandle",a,e,t,s),i=n("keys",a,e,t),u=n("values",a,e,t);return[await s.getHashTableById(r.id).import(i,u)]}case"LookupTableFind":case"LookupTableFindV2":{const r=n("tableHandle",a,e,t,s),i=n("keys",a,e,t),u=n("defaultValue",a,e,t);return[await s.getHashTableById(r.id).find(i,u)]}case"LookupTableSize":case"LookupTableSizeV2":{const r=n("tableHandle",a,e,t,s);return[s.getHashTableById(r.id).tensorSize()]}default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const du=(a,e,t,s=E)=>{switch(a.op){case"ResizeBilinear":{const r=n("images",a,e,t),i=n("size",a,e,t),u=n("alignCorners",a,e,t),o=n("halfPixelCenters",a,e,t);return[s.image.resizeBilinear(r,[i[0],i[1]],u,o)]}case"ResizeNearestNeighbor":{const r=n("images",a,e,t),i=n("size",a,e,t),u=n("alignCorners",a,e,t),o=n("halfPixelCenters",a,e,t);return[s.image.resizeNearestNeighbor(r,[i[0],i[1]],u,o)]}case"CropAndResize":{const r=n("image",a,e,t),i=n("boxes",a,e,t),u=n("boxInd",a,e,t),o=n("cropSize",a,e,t),p=n("method",a,e,t),m=n("extrapolationValue",a,e,t);return[s.image.cropAndResize(r,i,u,o,p,m)]}case"ImageProjectiveTransformV3":{const r=n("images",a,e,t),i=n("transforms",a,e,t),u=n("outputShape",a,e,t),o=n("fillValue",a,e,t),p=n("interpolation",a,e,t),m=n("fillMode",a,e,t);return[s.image.transform(r,i,p.toLowerCase(),m.toLowerCase(),o,u)]}default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const hu=(a,e,t,s=E)=>{switch(a.op){case"Equal":return[s.equal(n("a",a,e,t),n("b",a,e,t))];case"NotEqual":return[s.notEqual(n("a",a,e,t),n("b",a,e,t))];case"Greater":return[s.greater(n("a",a,e,t),n("b",a,e,t))];case"GreaterEqual":return[s.greaterEqual(n("a",a,e,t),n("b",a,e,t))];case"Less":return[s.less(n("a",a,e,t),n("b",a,e,t))];case"LessEqual":return[s.lessEqual(n("a",a,e,t),n("b",a,e,t))];case"LogicalAnd":return[s.logicalAnd(n("a",a,e,t),n("b",a,e,t))];case"LogicalNot":return[s.logicalNot(n("a",a,e,t))];case"LogicalOr":return[s.logicalOr(n("a",a,e,t),n("b",a,e,t))];case"Select":case"SelectV2":return[s.where(n("condition",a,e,t),n("a",a,e,t),n("b",a,e,t))];case"BitwiseAnd":return[s.bitwiseAnd(n("a",a,e,t),n("b",a,e,t))];default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const yu=(a,e,t,s=E)=>{switch(a.op){case"BatchMatMul":case"BatchMatMulV2":case"MatMul":return[s.matMul(n("a",a,e,t),n("b",a,e,t),n("transposeA",a,e,t),n("transposeB",a,e,t))];case"Einsum":return[s.einsum(n("equation",a,e,t),...n("tensors",a,e,t))];case"Transpose":return[s.transpose(n("x",a,e,t),n("perm",a,e,t))];case"_FusedMatMul":const[r,i]=n("fusedOps",a,e,t),u=r==="biasadd",o=i==="prelu",p=n("numArgs",a,e,t),m=n("leakyreluAlpha",a,e,t);if(u){if(o&&p!==2)throw new Error("Fused MatMul with BiasAdd and Prelu must have two extra arguments: bias and alpha.");if(!o&&p!==1)throw new Error("Fused MatMul with BiasAdd must have one extra argument: bias.")}const[l,c]=n("args",a,e,t);return[s.fused.matMul({a:n("a",a,e,t),b:n("b",a,e,t),transposeA:n("transposeA",a,e,t),transposeB:n("transposeB",a,e,t),bias:l,activation:i,preluActivationWeights:c,leakyreluAlpha:m})];case"MatrixBandPart":return[s.linalg.bandPart(n("a",a,e,t),n("numLower",a,e,t),n("numUpper",a,e,t))];default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const fu=(a,e,t,s=E)=>{switch(a.op){case"EuclideanNorm":return[s.euclideanNorm(n("x",a,e,t),n("axis",a,e,t),n("keepDims",a,e,t))];case"FusedBatchNorm":case"FusedBatchNormV2":return[s.batchNorm(n("x",a,e,t),n("mean",a,e,t),n("variance",a,e,t),n("offset",a,e,t),n("scale",a,e,t),n("epsilon",a,e,t))];case"FusedBatchNormV3":return[s.batchNorm(n("x",a,e,t),n("mean",a,e,t),n("variance",a,e,t),n("offset",a,e,t),n("scale",a,e,t),n("epsilon",a,e,t))];case"LRN":return[s.localResponseNormalization(n("x",a,e,t),n("radius",a,e,t),n("bias",a,e,t),n("alpha",a,e,t),n("beta",a,e,t))];case"Softmax":return[s.softmax(n("x",a,e,t))];case"LogSoftmax":return[s.logSoftmax(n("x",a,e,t))];default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const gu=(a,e,t,s=E)=>{switch(a.op){case"RaggedGather":{const{outputNestedSplits:r,outputDenseValues:i}=s.raggedGather(n("paramsNestedSplits",a,e,t),n("paramsDenseValues",a,e,t),n("indices",a,e,t),n("outputRaggedRank",a,e,t));return r.concat(i)}case"RaggedRange":{const{rtNestedSplits:r,rtDenseValues:i}=s.raggedRange(n("starts",a,e,t),n("limits",a,e,t),n("splits",a,e,t));return[r,i]}case"RaggedTensorToTensor":return[s.raggedTensorToTensor(n("shape",a,e,t),n("values",a,e,t),n("defaultValue",a,e,t),n("rowPartitionTensors",a,e,t),n("rowPartitionTypes",a,e,t))];default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Nu=(a,e,t,s=E)=>{switch(a.op){case"Max":{const o=n("axis",a,e,t),p=n("keepDims",a,e,t);return[s.max(n("x",a,e,t),o,p)]}case"Mean":{const o=n("axis",a,e,t),p=n("keepDims",a,e,t);return[s.mean(n("x",a,e,t),o,p)]}case"Min":{const o=n("axis",a,e,t),p=n("keepDims",a,e,t);return[s.min(n("x",a,e,t),o,p)]}case"Sum":{const o=n("axis",a,e,t),p=n("keepDims",a,e,t);return[s.sum(n("x",a,e,t),o,p)]}case"All":{const o=n("axis",a,e,t),p=n("keepDims",a,e,t);return[s.all(n("x",a,e,t),o,p)]}case"Any":{const o=n("axis",a,e,t),p=n("keepDims",a,e,t);return[s.any(n("x",a,e,t),o,p)]}case"ArgMax":{const o=n("axis",a,e,t);return[s.argMax(n("x",a,e,t),o)]}case"ArgMin":{const o=n("axis",a,e,t);return[s.argMin(n("x",a,e,t),o)]}case"Prod":{const o=n("axis",a,e,t),p=n("keepDims",a,e,t);return[s.prod(n("x",a,e,t),o,p)]}case"Cumprod":{const o=n("axis",a,e,t),p=n("exclusive",a,e,t),m=n("reverse",a,e,t);return[s.cumprod(n("x",a,e,t),o,p,m)]}case"Cumsum":{const o=n("axis",a,e,t),p=n("exclusive",a,e,t),m=n("reverse",a,e,t);return[s.cumsum(n("x",a,e,t),o,p,m)]}case"Bincount":const r=n("x",a,e,t),i=n("weights",a,e,t),u=n("size",a,e,t);return[s.bincount(r,i,u)];case"DenseBincount":{const o=n("x",a,e,t),p=n("weights",a,e,t),m=n("size",a,e,t),l=n("binaryOutput",a,e,t);return[s.denseBincount(o,p,m,l)]}default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const bu=(a,e,t,s=E)=>{switch(a.op){case"ConcatV2":case"Concat":{const r=n("n",a,e,t),i=n("axis",a,e,t);let u=n("tensors",a,e,t);return u=u.slice(0,r),[s.concat(u,i)]}case"Gather":{const r=n("x",a,e,t),i=n("indices",a,e,t);return[s.gather(r,s.cast(i,"int32"),0)]}case"GatherV2":{const r=n("axis",a,e,t),i=n("batchDims",a,e,t),u=n("x",a,e,t),o=n("indices",a,e,t);return[s.gather(u,s.cast(o,"int32"),r,i)]}case"Reverse":{const r=n("dims",a,e,t),i=[];for(let o=0;o<r.length;o++)r[o]&&i.push(o);const u=n("x",a,e,t);return[s.reverse(u,i)]}case"ReverseV2":{const r=n("axis",a,e,t),i=n("x",a,e,t);return[s.reverse(i,r)]}case"Slice":{const r=n("begin",a,e,t),i=n("size",a,e,t);return[s.slice(n("x",a,e,t),r,i)]}case"StridedSlice":{const r=n("begin",a,e,t),i=n("end",a,e,t),u=n("strides",a,e,t),o=n("beginMask",a,e,t),p=n("endMask",a,e,t),m=n("ellipsisMask",a,e,t),l=n("newAxisMask",a,e,t),c=n("shrinkAxisMask",a,e,t),d=n("x",a,e,t);return[s.stridedSlice(d,r,i,u,o,p,m,l,c)]}case"Pack":return P(()=>{const r=n("axis",a,e,t),i=n("tensors",a,e,t),u=i[0].shape,o=s.squeeze(i[0]).shape,p=i.map(m=>{const l=ae(m.shape,u);if(!l&&!ae(s.squeeze(m).shape,o))throw new Error("the input tensors shape does not match");return l?m:s.reshape(m,u)});return[s.stack(p,r)]});case"Unpack":{const r=n("axis",a,e,t),i=n("tensor",a,e,t);return s.unstack(i,r)}case"Tile":{const r=n("reps",a,e,t);return[s.tile(n("x",a,e,t),r)]}case"Split":case"SplitV":{const r=n("axis",a,e,t),i=n("numOrSizeSplits",a,e,t),u=n("x",a,e,t);return s.split(u,i,r)}case"ScatterNd":{const r=n("indices",a,e,t),i=n("values",a,e,t),u=n("shape",a,e,t);return[s.scatterND(r,i,u)]}case"GatherNd":{const r=n("x",a,e,t),i=n("indices",a,e,t);return[s.gatherND(r,i)]}case"SparseToDense":{const r=n("sparseIndices",a,e,t),i=n("outputShape",a,e,t),u=n("sparseValues",a,e,t),o=n("defaultValue",a,e,t);return[s.sparseToDense(r,u,i,u.dtype===o.dtype?o:s.cast(o,u.dtype))]}case"TensorScatterUpdate":{const r=n("indices",a,e,t),i=n("values",a,e,t),u=n("tensor",a,e,t);return[s.tensorScatterUpdate(u,r,i)]}default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Tu=(a,e,t,s=E)=>{switch(a.op){case"SparseFillEmptyRows":{const{outputIndices:r,outputValues:i,emptyRowIndicator:u,reverseIndexMap:o}=s.sparse.sparseFillEmptyRows(n("indices",a,e,t),n("values",a,e,t),n("denseShape",a,e,t),n("defaultValue",a,e,t));return[r,i,u,o]}case"SparseReshape":{const{outputIndices:r,outputShape:i}=s.sparse.sparseReshape(n("inputIndices",a,e,t),n("inputShape",a,e,t),n("newShape",a,e,t));return[r,i]}case"SparseSegmentMean":return[s.sparse.sparseSegmentMean(n("data",a,e,t),n("indices",a,e,t),n("segmentIds",a,e,t))];case"SparseSegmentSum":return[s.sparse.sparseSegmentSum(n("data",a,e,t),n("indices",a,e,t),n("segmentIds",a,e,t))];default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const wu=(a,e,t,s=E)=>{switch(a.op){case"FFT":return[s.fft(n("x",a,e,t))];case"IFFT":return[s.ifft(n("x",a,e,t))];case"RFFT":return[s.rfft(n("x",a,e,t))];case"IRFFT":return[s.irfft(n("x",a,e,t))];default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Su=(a,e,t,s=E)=>{switch(a.op){case"StaticRegexReplace":return[s.string.staticRegexReplace(n("input",a,e,t),n("pattern",a,e,t),n("rewrite",a,e,t),n("replaceGlobal",a,e,t))];case"StringNGrams":{const{nGrams:r,nGramsSplits:i}=s.string.stringNGrams(n("data",a,e,t),n("dataSplits",a,e,t),n("separator",a,e,t),n("nGramWidths",a,e,t),n("leftPad",a,e,t),n("rightPad",a,e,t),n("padWidth",a,e,t),n("preserveShortSequences",a,e,t));return[r,i]}case"StringSplit":{const{indices:r,values:i,shape:u}=s.string.stringSplit(n("input",a,e,t),n("delimiter",a,e,t),n("skipEmpty",a,e,t));return[r,i,u]}case"StringToHashBucketFast":return[s.string.stringToHashBucketFast(n("input",a,e,t),n("numBuckets",a,e,t))];default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ou=(a,e,t,s=E)=>{switch(a.op){case"Cast":return[s.cast(n("x",a,e,t),n("dtype",a,e,t))];case"ExpandDims":{const r=n("axis",a,e,t);return[s.expandDims(n("x",a,e,t),r)]}case"Squeeze":{const r=n("axis",a,e,t);return[s.squeeze(n("x",a,e,t),r)]}case"Reshape":return[s.reshape(n("x",a,e,t),n("shape",a,e,t))];case"EnsureShape":return[s.ensureShape(n("x",a,e,t),n("shape",a,e,t))];case"MirrorPad":return[s.mirrorPad(n("x",a,e,t),n("padding",a,e,t),n("mode",a,e,t))];case"PadV2":case"Pad":return[s.pad(n("x",a,e,t),n("padding",a,e,t),n("constantValue",a,e,t))];case"SpaceToBatchND":{const r=n("blockShape",a,e,t),i=n("paddings",a,e,t);return[s.spaceToBatchND(n("x",a,e,t),r,i)]}case"BatchToSpaceND":{const r=n("blockShape",a,e,t),i=n("crops",a,e,t);return[s.batchToSpaceND(n("x",a,e,t),r,i)]}case"DepthToSpace":{const r=n("blockSize",a,e,t),i=n("dataFormat",a,e,t).toUpperCase();return[s.depthToSpace(n("x",a,e,t),r,i)]}case"BroadcastTo":return[s.broadcastTo(n("x",a,e,t),n("shape",a,e,t))];case"BroadcastArgs":return[s.broadcastArgs(n("s0",a,e,t),n("s1",a,e,t))];default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function at(a,e,t,s,r=P){const i=((u,o,p)=>{switch(u.category){case"arithmetic":return r(()=>Qo(u,o,p));case"basic_math":return r(()=>Mo(u,o,p));case"control":return nu(u,o,p);case"convolution":return r(()=>iu(u,o,p));case"creation":return r(()=>ou(u,o,p));case"dynamic":return uu(u,o,p);case"evaluation":return r(()=>pu(u,o,p));case"image":return r(()=>du(u,o,p));case"graph":return r(()=>mu(u,o,p));case"logical":return r(()=>hu(u,o,p));case"matrices":return r(()=>yu(u,o,p));case"normalization":return r(()=>fu(u,o,p));case"ragged":return r(()=>gu(u,o,p));case"reduction":return r(()=>Nu(u,o,p));case"slice_join":return r(()=>bu(u,o,p));case"sparse":return r(()=>Tu(u,o,p));case"spectral":return r(()=>wu(u,o,p));case"string":return r(()=>Su(u,o,p));case"transformation":return r(()=>Ou(u,o,p));case"hash_table":return cu(u,o,p,s);case"custom":const m=$t(u.op);if(m&&m.customExecutor)return m.customExecutor(new Yo(u,o,p));throw TypeError(`Custom op ${u.op} is not registered.`);default:throw TypeError(`Unknown op '${u.op}'. File an issue at https://github.com/tensorflow/tfjs/issues so we can add it, or register a custom execution with tf.registerOp()`)}})(a,e,t);return de(i)?i.then(u=>[].concat(u)):[].concat(i)}class st{constructor(e={},t={},s={},r={},i){this.weightMap=e,this.tensorArrayMap=t,this.tensorListMap=s,this.functionMap=r,this.parseNodeNameCache=i,this.rootContext={id:0,frameName:"",iterationId:0},this.contexts=[this.rootContext],this.lastId=0,this.generateCurrentContextIds()}newFrame(e,t){return{id:e,frameName:t,iterationId:0}}set currentContext(e){this.contexts!==e&&(this.contexts=e,this.generateCurrentContextIds())}get currentContext(){return this.contexts}get currentContextId(){return this._currentContextIds[0]}get currentContextIds(){return this._currentContextIds}generateCurrentContextIds(){const e=[];for(let t=0;t<this.contexts.length-1;t++){const s=this.contexts.slice(0,this.contexts.length-t);e.push(this.contextIdforContexts(s))}e.push(""),this._currentContextIds=e}contextIdforContexts(e){return e?e.map(t=>t.id===0&&t.iterationId===0?"":`${t.frameName}-${t.iterationId}`).join("/"):""}enterFrame(e){this.contexts&&(this.lastId++,this.contexts=this.contexts.slice(),this.contexts.push(this.newFrame(this.lastId,e)),this._currentContextIds.unshift(this.contextIdforContexts(this.contexts)))}exitFrame(){if(this.contexts&&this.contexts.length>1)this.contexts=this.contexts.slice(),this.contexts.splice(-1),this.currentContextIds.shift();else throw new Error("Cannot exit frame, the context is empty")}nextIteration(){if(this.contexts&&this.contexts.length>0){this.contexts=this.contexts.slice(),this.lastId++;const e=Object.assign({},this.contexts[this.contexts.length-1]);e.iterationId+=1,e.id=this.lastId,this.contexts.splice(-1,1,e),this._currentContextIds.splice(0,1,this.contextIdforContexts(this.contexts))}else throw new Error("Cannot increase frame iteration, the context is empty")}getWeight(e){return this.weightMap[e]}addTensorArray(e){this.tensorArrayMap[e.id]=e}getTensorArray(e){return this.tensorArrayMap[e]}addTensorList(e){this.tensorListMap[e.id]=e}getTensorList(e){return this.tensorListMap[e]}dispose(e){for(const t in this.tensorArrayMap)this.tensorArrayMap[t].clearAndClose(e);for(const t in this.tensorListMap)this.tensorListMap[t].clearAndClose(e)}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function rt(a,e,t,s){const r=new Set,i=[];let u=null,o=null;const p=new Set,m=new Set(Object.keys(a).map(d=>$(d)[0]));s=s||[];const l=new Set(s.map(d=>$(d.name)[0])),c=[...e];for(;c.length>0;){const d=c.pop();if((W(d)||Du(d)||zu(d))&&u==null&&(u=d,o=u.children.map(h=>h.name).filter(h=>r.has(h))),r.add(d.name),t[d.name]==null&&!m.has(d.name)&&!l.has(d.name)){if(d.inputs.length===0){i.push(d.name);continue}d.inputs.forEach(h=>{p.has(h.name)||(p.add(h.name),c.push(h))})}}return{inputs:a,outputs:e,usedNodes:r,missingInputs:i,dynamicNode:u,syncInputs:o}}function vu(a,e){const{usedNodes:t,inputs:s}=e,r=Object.keys(s).map(y=>$(y)[0]).map(y=>a.nodes[y]),i=a.initNodes||[],u=y=>t.has(typeof y=="string"?y:y.name);function o(y){return[...new Map(y.map(b=>[b.name,b])).values()]}const p=o([...r,...a.weights,...i]).filter(u),m=o([...p,...Object.values(a.nodes)]).filter(u),l=new Map(m.map(y=>[y.name,y])),c={};for(const y of m){c[y.name]=c[y.name]||0;for(const b of y.children)u(b)||(c[b.name]=Number.POSITIVE_INFINITY),c[b.name]=(c[b.name]||0)+1}const d=Object.entries(c).filter(([,y])=>y===0).map(([y])=>y),h=[...d];for(;d.length>0;){const y=d.pop(),b=l.get(y);for(const _ of b.children.filter(u))--c[_.name]===0&&(h.push(_.name),d.push(_.name))}const N=h.map(y=>l.get(y)),f=_u(N,p);return Au(f,p),f}function _u(a,e){const t=new Map(a.map(u=>[u.name,u])),s=e.map(u=>u.name),r=new Set(s);for(;s.length>0;){const u=s.pop(),o=t.get(u);for(const p of o.children)!t.has(p.name)||r.has(p.name)||(r.add(p.name),s.push(p.name))}return a.filter(u=>r.has(u.name))}class ue extends Error{constructor(e){super(`NodesExecutionOrderError: ${e}`)}}function Au(a,e){const t=new Map(a.map((o,p)=>[o.name,p])),s=new Set(e.map(o=>o.name)),r=o=>s.has(typeof o=="string"?o:o.name),i=new Set(a.map(o=>o.name)),u=o=>i.has(typeof o=="string"?o:o.name);for(const o of a){for(const p of o.children.filter(u)){if(!t.has(p.name))throw new ue(`Child ${p.name} of node ${o.name} is unreachable.`);if(t.get(o.name)>t.get(p.name))throw new ue(`Node ${o.name} is scheduled to run after its child ${p.name}.`)}if(!r(o))for(const p of o.inputs){if(!t.has(p.name))throw new ue(`Input ${p.name} of node ${o.name} is unreachable.`);if(t.get(p.name)>t.get(o.name))throw new ue(`Node ${o.name} is scheduled to run before its input ${p.name}.`)}}}function Eu(a){const e=new Map(a.map((o,p)=>[o.name,p])),t=Number.MAX_SAFE_INTEGER,s=a.map((o,p)=>W(o)?t:p),r=o=>{const p=s[e.get(o.name)];return p??-1},i=a.map((o,p)=>o.children.map(r).reduce((m,l)=>Math.max(m,l),s[p])),u=new Map;for(let o=0;o<a.length;++o){const p=i[o];if(p===t)continue;const m=a[o],l=a[p];u.has(l.name)||u.set(l.name,[]),u.get(l.name).push(m)}return u}const Iu=new Set(["Switch","Merge","Enter","Exit","NextIteration","StatelessIf","StatelessWhile","if","While"]),ku=new Set(["NonMaxSuppressionV2","NonMaxSuppressionV3","NonMaxSuppressionV5","Where"]),$u=new Set(["HashTable","HashTableV2","LookupTableImport","LookupTableImportV2","LookupTableFind","LookupTableFindV2","LookupTableSize","LookupTableSizeV2"]);function W(a){return Iu.has(a.op)}function Du(a){return ku.has(a.op)}function zu(a){return $u.has(a.op)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class fe{get weightIds(){return this.parent?this.parent.weightIds:this._weightIds}get functionExecutorMap(){return this.parent?this.parent.functionExecutorMap:this._functionExecutorMap}get weightMap(){return this.parent?this.parent.weightMap:this._weightMap}set weightMap(e){const t=Object.keys(e).map(s=>e[s].map(r=>r.id));this._weightIds=[].concat(...t),this._weightMap=e}set resourceManager(e){this._resourceManager=e}get inputs(){return this._inputs.map(e=>({name:e.name,shape:e.attrParams.shape?e.attrParams.shape.value:void 0,dtype:e.attrParams.dtype?e.attrParams.dtype.value:void 0}))}get outputs(){return this._outputs.map(e=>({name:e.name,shape:e.attrParams.shape?e.attrParams.shape.value:void 0,dtype:e.attrParams.dtype?e.attrParams.dtype.value:void 0}))}get inputNodes(){return this._inputs.map(e=>e.signatureKey||e.name)}get outputNodes(){return this._outputs.map(e=>{const t=e.signatureKey||e.name;return e.defaultOutput?`${t}:${e.defaultOutput}`:t})}get functions(){return Object.keys(this._functions).reduce((e,t)=>(e[t]=this._functions[t].signature,e),{})}constructor(e,t){this.graph=e,this.parent=t,this.compiledMap=new Map,this.parseNodeNameCache=new Map,this._weightMap={},this.SEPARATOR=",",this._functions={},this._functionExecutorMap={},this.keepIntermediateTensors=!1,this._outputs=e.outputs,this._inputs=e.inputs,this._initNodes=e.initNodes,this._signature=e.signature,this._functions=e.functions,e.functions!=null&&Object.keys(e.functions).forEach(s=>{this._functionExecutorMap[s]=new fe(e.functions[s],this)})}getCompilationKey(e,t){const s=e.map(i=>i.name).sort(),r=t.map(i=>i.name).sort();return s.join(this.SEPARATOR)+"--"+r.join(this.SEPARATOR)}compile(e,t){const s=rt(e,t,this.weightMap,this._initNodes),{missingInputs:r,dynamicNode:i,syncInputs:u}=s;if(i!=null)throw new Error(`This execution contains the node '${i.name}', which has the dynamic op '${i.op}'. Please use model.executeAsync() instead. Alternatively, to avoid the dynamic ops, specify the inputs [${u}]`);if(r.length>0){const m=t.map(c=>c.name),l=Object.keys(e);throw new Error(`Cannot compute the outputs [${m}] from the provided inputs [${l}]. Missing the following inputs: [${r}]`)}const o=vu(this.graph,s),p=Eu(o);return{orderedNodes:o,nodeLiveUntilMap:p}}cloneAndKeepTensor(e){if(e==null)return null;const t=e.clone();return R(t),t}cloneTensorList(e){return e?e.map(s=>this.cloneAndKeepTensor(s)):null}cloneTensorMap(e){return Object.fromEntries(Object.entries(e).map(([t,s])=>[t,this.cloneTensorList(s)]))}execute(e,t){this.disposeIntermediateTensors(),e=this.mapInputs(e);const s=Object.keys(e).sort();this.checkInputs(e),this.checkInputShapeAndType(e),t=this.mapOutputs(t),this.checkOutputs(t);const r=s.map(d=>this.graph.nodes[$(d)[0]]),i=t.map(d=>$(d)[0]),u=new Set(i);let o=i.map(d=>this.graph.nodes[d]);o.length===0&&(o=this._outputs);const p=this.getCompilationKey(r,o);let m=this.compiledMap.get(p);m==null&&(m=this.compile(e,o),this.compiledMap.set(p,m));try{this.keepIntermediateTensors=Q().getBool("KEEP_INTERMEDIATE_TENSORS")}catch(d){this.keepIntermediateTensors=!1,console.warn(d.message)}const l={},c={};return P(()=>{const d=new st(this.weightMap,l,c,this.functionExecutorMap,this.parseNodeNameCache),h=Object.assign({},this.weightMap);this.keepIntermediateTensors&&(this.clonedTensorsMap=this.cloneTensorMap(this.weightMap)),Object.keys(e).forEach(b=>{const[_,k]=$(b,d),w=[];w[k]=e[b],h[_]=w,this.keepIntermediateTensors&&(this.clonedTensorsMap[_]=this.cloneTensorList(w))});const N=this.getFrozenTensorIds(h),{orderedNodes:f,nodeLiveUntilMap:y}=m;for(const b of f){if(h[b.name])continue;const _=at(b,h,d,this._resourceManager);if(de(_))throw new Error(`The execution of the op '${b.op}' returned a promise. Please use model.executeAsync() instead.`);h[b.name]=_,this.keepIntermediateTensors&&(this.clonedTensorsMap[b.name]=this.cloneTensorList(_)),this.checkTensorForDisposalWithNodeLiveUntilInfo(b,h,d,N,u,y.get(b.name))}return this.parent==null&&d.dispose(N),t.map(b=>A(b,h,d))})}getFrozenTensorIds(e){const t=[].concat.apply([],Object.keys(e).map(s=>e[s]).map(s=>s.map(r=>r.id)));return new Set(t)}checkTensorForDisposal(e,t,s,r,i,u,o){if(!(W(t)||u.has(e))){for(const p of s[e])p!=null&&(o[p.id]=(o[p.id]||0)+t.children.length);for(const p of t.inputs){if(W(p))continue;const m=Ye(p.name,s,r);if(m!=null)for(const l of m){if(!l||l.kept||i.has(l.id))continue;const c=o[l.id];c===1?(l.dispose(),delete o[l.id]):c!=null&&o[l.id]--}}}}checkTensorForDisposalWithNodeLiveUntilInfo(e,t,s,r,i,u){function o(p){return W(p)||i.has(p.name)}if(!(W(e)||u==null))for(const p of u){if(o(p))continue;const m=Ye(p.name,t,s);for(const l of m)!l||l.kept||r.has(l.id)||l.dispose()}}async executeAsync(e,t){return this._executeAsync(e,t)}disposeIntermediateTensors(){this.clonedTensorsMap&&(Object.values(this.clonedTensorsMap).forEach(e=>{for(const t of e)t&&!t.isDisposed&&t.dispose()}),this.clonedTensorsMap=null)}getIntermediateTensors(){return this.clonedTensorsMap}async _executeAsync(e,t,s=!1,r={},i={}){this.disposeIntermediateTensors(),s||(e=this.mapInputs(e),this.checkInputs(e),this.checkInputShapeAndType(e),t=this.mapOutputs(t),this.checkOutputs(t));try{this.keepIntermediateTensors=Q().getBool("KEEP_INTERMEDIATE_TENSORS")}catch(d){this.keepIntermediateTensors=!1,console.warn(d.message)}const u=new st(this.weightMap,r,i,this.functionExecutorMap,this.parseNodeNameCache);this.keepIntermediateTensors&&(this.clonedTensorsMap=this.cloneTensorMap(this.weightMap));const o=await this.executeWithControlFlow(e,u,t,s),p=t.map(d=>A(d,o,u)),m=p.map(d=>d.id),l=Object.keys(e).map(d=>e[d].id),c=new Set([...m,...l,...this.weightIds]);return Object.values(o).forEach(d=>{d.forEach(h=>{h&&!h.isDisposed&&!c.has(h.id)&&h.dispose()})}),this.parent==null&&u.dispose(c),p}async executeFunctionAsync(e,t,s){const r=e.reduce((i,u,o)=>(i[this.inputs[o].name]=u,i),{});return this._executeAsync(r,this.outputNodes,!0,t,s)}async executeWithControlFlow(e,t,s,r){const i=Object.keys(e),u=i.map(w=>this.graph.nodes[$(w)[0]]),o=s.map(w=>$(w)[0]),p=new Set(o);let m=o.map(w=>this.graph.nodes[w]);m.length===0&&(m=this._outputs);const{usedNodes:l,missingInputs:c,dynamicNode:d,syncInputs:h}=rt(e,m,this.weightMap,this._initNodes),N=[...u,...this.graph.weights,...this._initNodes||[]].map(w=>({node:w,contexts:t.currentContext})),f=Object.assign({},this.weightMap);Object.keys(e).forEach(w=>{const[I,D]=$(w),z=[];z[D]=e[w],f[I]=z});const y={},b=this.getFrozenTensorIds(f),_={};for(;N.length>0;){const w=this.processStack(u,N,t,f,_,b,p,y,l);await Promise.all(w)}d==null&&!r&&console.warn("This model execution did not contain any nodes with control flow or dynamic output shapes. You can use model.execute() instead.");const k=m.filter(w=>!W(w)&&!A(w.name,f,t)).map(w=>w.name);if(k.length>0){let w="";throw d!=null&&(w=`Alternatively, to avoid the dynamic ops, use model.execute() and specify the inputs [${h}]`),new Error(`Cannot compute the outputs [${k}] from the provided inputs [${i}]. Consider providing the following inputs: [${c}]. ${w}`)}return f}processStack(e,t,s,r,i,u,o,p,m){const l=[];for(;t.length>0;){const c=t.pop();s.currentContext=c.contexts;let d="";if(c.node.op==="Enter"&&n("isConstant",c.node,r,s)&&([d]=j(c.node.name,s)),r[c.node.name]==null){const h=at(c.node,r,s,this._resourceManager);d||([d]=j(c.node.name,s));const N=s.currentContext;de(h)?l.push(h.then(f=>(r[d]=f,this.keepIntermediateTensors&&(this.clonedTensorsMap[d]=this.cloneTensorList(f)),s.currentContext=N,this.checkTensorForDisposal(d,c.node,r,s,u,o,p),this.processChildNodes(c.node,t,s,r,i,m),f))):(r[d]=h,this.keepIntermediateTensors&&(this.clonedTensorsMap[d]=this.cloneTensorList(h)),this.checkTensorForDisposal(d,c.node,r,s,u,o,p),this.processChildNodes(c.node,t,s,r,i,m))}else this.processChildNodes(c.node,t,s,r,i,m)}return l}processChildNodes(e,t,s,r,i,u){e.children.forEach(o=>{const[p]=j(o.name,s);i[p]||!u.has(o.name)||(o.op==="Merge"?o.inputNames.some(m=>!!A(m,r,s))&&(i[p]=!0,t.push({contexts:s.currentContext,node:o})):o.inputNames.every(m=>!!A(m,r,s))&&(i[p]=!0,t.push({contexts:s.currentContext,node:o})))})}dispose(){Object.keys(this.weightMap).forEach(e=>this.weightMap[e].forEach(t=>t.dispose()))}checkInputShapeAndType(e){Object.keys(e).forEach(t=>{const s=e[t],[r]=$(t),i=this.graph.nodes[r];if(i.attrParams.shape&&i.attrParams.shape.value){const u=i.attrParams.shape.value,o=u.length===s.shape.length&&s.shape.every((p,m)=>u[m]===-1||u[m]===p);S(o,()=>`The shape of dict['${i.name}'] provided in model.execute(dict) must be [${u}], but was [${s.shape}]`)}i.attrParams.dtype&&i.attrParams.dtype.value&&S(s.dtype===i.attrParams.dtype.value,()=>`The dtype of dict['${i.name}'] provided in model.execute(dict) must be ${i.attrParams.dtype.value}, but was ${s.dtype}`)})}mapInputs(e){var t,s;const r={};for(const i in e){const u=(s=(t=this._signature)===null||t===void 0?void 0:t.inputs)===null||s===void 0?void 0:s[i];u!=null?r[u.name]=e[i]:r[i]=e[i]}return r}checkInputs(e){const t=Object.keys(e).filter(s=>{const[r]=$(s);return this.graph.nodes[r]==null});if(t.length>0)throw new Error(`The dict provided in model.execute(dict) has keys: [${t}] that are not part of graph`)}mapOutputs(e){return e.map(t=>{var s,r;const i=(r=(s=this._signature)===null||s===void 0?void 0:s.outputs)===null||r===void 0?void 0:r[t];return i!=null?i.name:t},{})}checkOutputs(e){e.forEach(t=>{const[s]=$(t);if(!this.graph.nodes[s])throw new Error(`The output '${t}' is not found in the graph`)})}}class Lu{constructor(e={},t={}){this.hashTableNameToHandle=e,this.hashTableMap=t}addHashTable(e,t){this.hashTableNameToHandle[e]=t.handle,this.hashTableMap[t.id]=t}getHashTableHandleByName(e){return this.hashTableNameToHandle[e]}getHashTableById(e){return this.hashTableMap[e]}dispose(){for(const e in this.hashTableMap)this.hashTableMap[e].clearAndClose(),delete this.hashTableMap[e];for(const e in this.hashTableNameToHandle)this.hashTableNameToHandle[e].dispose(),delete this.hashTableNameToHandle[e]}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Cu="?tfjs-format=file",Vu="model.json";class Lt{get modelVersion(){return this.version}get inputNodes(){return this.executor.inputNodes}get outputNodes(){return this.executor.outputNodes}get inputs(){return this.executor.inputs}get outputs(){return this.executor.outputs}get weights(){return this.executor.weightMap}get metadata(){return this.artifacts.userDefinedMetadata}get modelSignature(){return this.signature}get modelStructuredOutputKeys(){return this.structuredOutputKeys}constructor(e,t={},s=Et){this.modelUrl=e,this.loadOptions=t,this.version="n/a",this.io=s,t==null&&(this.loadOptions={}),this.resourceManager=new Lu}findIOHandler(){const e=this.modelUrl;if(e.load!=null)this.handler=e;else if(this.loadOptions.requestInit!=null)this.handler=this.io.browserHTTPRequest(e,this.loadOptions);else{const t=this.io.getLoadHandlers(e,this.loadOptions);if(t.length===0)t.push(this.io.browserHTTPRequest(e,this.loadOptions));else if(t.length>1)throw new Error(`Found more than one (${t.length}) load handlers for URL '${[e]}'`);this.handler=t[0]}}load(){if(this.findIOHandler(),this.handler.load==null)throw new Error("Cannot proceed with model loading because the IOHandler provided does not have the `load` method implemented.");const e=this.handler.load();return de(e)?e.then(t=>t.getWeightStream==null?this.loadSync(t):this.loadStreaming(t)):this.loadSync(e)}loadSync(e){const t=this.io.decodeWeights(e.weightData,e.weightSpecs);return this.loadWithWeightMap(e,t)}async loadStreaming(e){if(e.getWeightStream==null)throw new Error("Model artifacts missing streamWeights function");const t=await Nt(e.getWeightStream(),e.weightSpecs);return this.loadWithWeightMap(e,t)}loadWithWeightMap(e,t){this.artifacts=e;const s=this.artifacts.modelTopology;let r=this.artifacts.signature;if(this.artifacts.userDefinedMetadata!=null){const i=this.artifacts.userDefinedMetadata;i.signature!=null&&(r=i.signature),i.structuredOutputKeys!=null&&(this.structuredOutputKeys=i.structuredOutputKeys)}if(this.signature=r,this.version=`${s.versions.producer}.${s.versions.minConsumer}`,this.executor=new fe(Qe.Instance.transformGraph(s,this.signature)),this.executor.weightMap=this.convertTensorMapToTensorsMap(t),this.executor.resourceManager=this.resourceManager,e.modelInitializer!=null&&e.modelInitializer.node!=null){const i=Qe.Instance.transformGraph(e.modelInitializer);this.initializer=new fe(i),this.initializer.weightMap=this.executor.weightMap,this.initializer.resourceManager=this.resourceManager,this.initializerSignature=e.initializerSignature}return!0}async save(e,t){if(typeof e=="string"){const s=this.io.getSaveHandlers(e);if(s.length===0)throw new Error(`Cannot find any save handlers for URL '${e}'`);if(s.length>1)throw new Error(`Found more than one (${s.length}) save handlers for URL '${e}'`);e=s[0]}if(e.save==null)throw new Error("GraphModel.save() cannot proceed because the IOHandler provided does not have the `save` attribute defined.");return e.save(this.artifacts)}addStructuredOutputNames(e){if(this.structuredOutputKeys){const t=e instanceof Z?[e]:e,s={};return t.forEach((r,i)=>s[this.structuredOutputKeys[i]]=r),s}return e}predict(e,t){const s=this.execute(e,this.outputNodes);return this.addStructuredOutputNames(s)}async predictAsync(e,t){const s=await this.executeAsync(e,this.outputNodes);return this.addStructuredOutputNames(s)}normalizeInputs(e){var t;if(!(e instanceof Z)&&!Array.isArray(e)){const i=(t=this.signature)===null||t===void 0?void 0:t.inputs;if(i!=null)for(const u in i){const o=i[u];o.resourceId!=null&&(e[u]=this.resourceIdToCapturedInput[o.resourceId])}return e}e=Array.isArray(e)?e:[e];const s=Object.keys(this.resourceIdToCapturedInput).length;if(e.length+s!==this.inputNodes.length)throw new Error(`Input tensor count mismatch, the graph model has ${this.inputNodes.length-s} non-resource placeholders, while there are ${e.length} input tensors provided.`);let r=0;return this.inputNodes.reduce((i,u)=>{var o,p,m;const l=(m=(p=(o=this.signature)===null||o===void 0?void 0:o.inputs)===null||p===void 0?void 0:p[u])===null||m===void 0?void 0:m.resourceId;return l!=null?i[u]=this.resourceIdToCapturedInput[l]:i[u]=e[r++],i},{})}normalizeOutputs(e){return e=e||this.outputNodes,Array.isArray(e)?e:[e]}executeInitializerGraph(){return this.initializer==null?[]:this.initializerSignature==null?this.initializer.execute({},[]):this.initializer.execute({},Object.keys(this.initializerSignature.outputs))}async executeInitializerGraphAsync(){return this.initializer==null?[]:this.initializerSignature==null?this.initializer.executeAsync({},[]):this.initializer.executeAsync({},Object.keys(this.initializerSignature.outputs))}setResourceIdToCapturedInput(e){if(this.resourceIdToCapturedInput={},this.initializerSignature){const t=this.initializerSignature.outputs,s=Object.keys(t);for(let r=0;r<s.length;r++){const i=s[r],u=t[i];this.resourceIdToCapturedInput[u.resourceId]=e[r]}}}execute(e,t){this.resourceIdToCapturedInput==null&&this.setResourceIdToCapturedInput(this.executeInitializerGraph()),e=this.normalizeInputs(e),t=this.normalizeOutputs(t);const s=this.executor.execute(e,t);return s.length>1?s:s[0]}async executeAsync(e,t){this.resourceIdToCapturedInput==null&&this.setResourceIdToCapturedInput(await this.executeInitializerGraphAsync()),e=this.normalizeInputs(e),t=this.normalizeOutputs(t);const s=await this.executor.executeAsync(e,t);return s.length>1?s:s[0]}getIntermediateTensors(){return this.executor.getIntermediateTensors()}disposeIntermediateTensors(){this.executor.disposeIntermediateTensors()}convertTensorMapToTensorsMap(e){return Object.keys(e).reduce((t,s)=>(t[s]=[e[s]],t),{})}dispose(){this.executor.dispose(),this.initializer&&(this.initializer.dispose(),this.resourceIdToCapturedInput&&bn(this.resourceIdToCapturedInput)),this.resourceManager.dispose()}}async function Bu(a,e={},t=Et){if(a==null)throw new Error("modelUrl in loadGraphModel() cannot be null. Please provide a url or an IOHandler that loads the model");e==null&&(e={}),e.fromTFHub&&typeof a=="string"&&(a=Pu(a));const s=new Lt(a,e,t);return await s.load(),s}function Hu(a){if(a==null)throw new Error("modelUrl in loadGraphModelSync() cannot be null. Please provide model artifacts or an IOHandler that loads the model");let e;if(a instanceof Array){const[s,r]=a;if(!s)throw new Error("modelJSON must be the first element of the array");if(!r||!(r instanceof ArrayBuffer))throw new Error("An ArrayBuffer of weights must be the second element of the array");if(!("modelTopology"in s))throw new Error("Model JSON is missing 'modelTopology'");if(!("weightsManifest"in s))throw new Error("Model JSON is missing 'weightsManifest'");const i=Tt(s.weightsManifest),u=bt(s,i,r);e=he(u)}else if("load"in a)e=a;else if("modelTopology"in a&&"weightSpecs"in a&&"weightData"in a)e=he(a);else throw new Error("Unknown model format");const t=new Lt(e);return t.load(),t}function Pu(a){return a.endsWith("/")||(a=a+"/"),`${a}${Vu}${Cu}`}export{pi as A,li as B,di as C,yi as D,gi as E,bi as F,wi as G,vt as H,Si as I,Oi as J,vi as K,Ai as L,Ei as M,_t as N,$i as O,zi as P,Ci as Q,je as R,Fi as S,xi as T,Bi as U,Lt as V,Bu as W,Hu as X,Ru as Y,xu as Z,qi as a,ju as b,wn as c,On as d,_n as e,mo as f,En as g,kn as h,Et as i,Dn as j,Cn as k,zn as l,Pn as m,Fn as n,xn as o,Bn as p,Wn as q,Un as r,Kn as s,Xn as t,Yn as u,Mn as v,ti as w,si as x,ni as y,oi as z};
