// Headless checks for the polyhedra clustered view.
//
//     node tests/web/test_polyhedra_cluster.mjs
//
// The layout engine is shared with the surfaces and is covered by
// test_cluster_cache.mjs. What is specific here, and what this checks,
// is the wiring: that the view is handed the POLYHEDRA sprite sheet and
// the polyhedra similarity vector rather than defaulting to the
// surfaces' -- a default that would have loaded the wrong sheet and
// clustered on fields the solids do not have, both of which fail
// silently as an empty or arbitrary field rather than as an error.
import fs from 'node:fs';
let W=840,H=740,budget=0,loads=0;
globalThis.requestAnimationFrame=(cb)=>{if(budget-->0)cb();return 1;};
globalThis.cancelAnimationFrame=()=>{};
globalThis.devicePixelRatio=1;
globalThis.window=globalThis;
globalThis.addEventListener=()=>{};
globalThis.requestIdleCallback=(cb)=>cb();
globalThis.fetch=(u)=>{
  const s=String(u);
  loads++;
  if(s.includes('polyhedra-atlas.json'))
    return Promise.resolve({ok:true,json:()=>Promise.resolve({cell:128,cols:22,rows:22,tiles:{}})});
  return Promise.resolve({ok:false});
};
globalThis.Image=class{ set src(_v){ Promise.resolve().then(()=>this.onload&&this.onload()); } };
const ctx=new Proxy({},{get:()=>()=>{}});
const canvas={width:0,height:0,getContext:()=>ctx,addEventListener:()=>{},
  getBoundingClientRect:()=>({width:W,height:H}),style:{}};
const HERE=new URL('.', import.meta.url);
const CV=await import(new URL('../../web/js/cluster-view.js',HERE).href);
const P =await import(new URL('../../web/js/cluster-polyhedra.js',HERE).href);
const all=JSON.parse(fs.readFileSync(
  new URL('../../data/polyhedra/index.json',HERE),'utf8')).entries;
const flush=()=>new Promise(r=>setTimeout(r,0));
let fail=0; const ok=(c,m)=>{if(!c){fail++;console.log('  FAIL '+m);}else console.log('  ok   '+m);};

const view=new CV.ClusterView(canvas,all,{atlas:'polyhedra',vector:P.polyhedronVector,
  thumbUrl:(s)=>s+'.png'});
async function settle(e){ budget=250; const t=Date.now(); view.show(e); await flush(); return Date.now()-t; }

console.log('1. it lays the whole catalogue out');
const t1=await settle(all);
ok(!!view.layout,'layout built');
ok(view.entries.length===471,`471 solids (${view.entries.length})`);
ok(view.view.k>0&&isFinite(view.view.k),`framed (k=${view.view.k.toFixed(2)})`);
console.log(`   cold ${t1}ms`);

console.log('2. it uses the POLYHEDRA sheet, not the surfaces one');
ok(!!view.atlas,'an atlas was loaded');
ok(view.atlasName==='polyhedra',`atlasName=${view.atlasName}`);

console.log('3. the cache works here too');
await settle(all.filter(x=>x.families.includes('johnson')));
const t3=await settle(all);
ok(t3*4<t1||t3<5,`returning is fast (${t3}ms vs ${t1}ms)`);

console.log('4. filtered subsets');
for(const tag of ['johnson','compound','platonic','archimedean']){
  const e=all.filter(x=>(x.families||[]).includes(tag));
  await settle(e);
  const n=e.length;
  let off=0;
  for(let i=0;i<n;i++){
    const x=view.layout.x[i]*view.view.k+view.view.x;
    const y=view.layout.y[i]*view.view.k+view.view.y;
    if(x<0||x>W||y<0||y>H)off++;
  }
  ok(off===0,`${tag} (n=${n}): nothing off-canvas`);
}
console.log(fail?`\nRESULT: ${fail} FAILURE(S)`:'\nRESULT: OK');
process.exit(fail?1:0);
