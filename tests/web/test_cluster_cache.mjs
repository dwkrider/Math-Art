// Headless checks for the clustered view's layout cache and tile gating.
//
//     node tests/web/test_cluster_cache.mjs
//
// Run by tests/test_web.py when node is available, and skipped with a
// note when it is not, like the rest of the JS checks there.
//
// These two behaviours are invisible to every other gate. A broken cache
// is not a wrong picture, it is the right picture arrived at slowly, and
// a broken gate is the right picture arrived at in the wrong order --
// both look fine in a screenshot. What they need is a stopwatch and a
// stubbed DOM, which is what this is.
import fs from 'node:fs';
let W=840,H=740,budget=0;
let atlasDelay = 0, atlasLoads = 0;
globalThis.requestAnimationFrame=(cb)=>{if(budget-->0)cb();return 1;};
globalThis.cancelAnimationFrame=()=>{};
globalThis.devicePixelRatio=1;
globalThis.window=globalThis;
globalThis.addEventListener=()=>{};
// A fake atlas that resolves after `atlasDelay` microtask turns.
globalThis.fetch=()=>Promise.resolve({ok:true,json:()=>Promise.resolve(
  {cell:128,cols:22,rows:22,tiles:{}})});
globalThis.Image=class{
  set src(_v){ atlasLoads++;
    let p=Promise.resolve();
    for(let i=0;i<atlasDelay;i++)p=p.then(()=>{});
    p.then(()=>this.onload&&this.onload()); }
};
const ctx=new Proxy({},{get:()=>()=>{}});
const canvas={width:0,height:0,getContext:()=>ctx,addEventListener:()=>{},
  getBoundingClientRect:()=>({width:W,height:H}),style:{}};
const HERE = new URL('.', import.meta.url);
const M = await import(new URL('../../web/js/cluster-view.js', HERE).href);
const all = JSON.parse(fs.readFileSync(
  new URL('../../data/surfaces/index.json', HERE), 'utf8')).entries;
const fam=(f)=>all.filter(e=>e.primary_family===f);
const flush=()=>new Promise(r=>setTimeout(r,0));

let fail=0; const ok=(c,m)=>{if(!c){fail++;console.log('  FAIL '+m);}else console.log('  ok   '+m);};

const view=new M.ClusterView(canvas,all,{thumbUrl:s=>s+'.png'});
async function settle(entries){
  // Budget set BEFORE show(), so the frames are available to the gated
  // _run() when the tile promise resolves and it finally starts.
  budget = 200;
  const t = Date.now();
  view.show(entries);
  await flush();
  return Date.now() - t;
}
const snap=()=>({x:Float64Array.from(view.layout.x),y:Float64Array.from(view.layout.y)});
const same=(a,b)=>a.x.length===b.x.length&&a.x.every((v,i)=>v===b.x[i])&&a.y.every((v,i)=>v===b.y[i]);

console.log('1. layout does not start before the tiles arrive');
atlasDelay = 5;
M.preloadAtlas();                      // page-load preload
view.show(fam('quadric'));
ok(view._waiting === true, 'waiting flag set immediately on show()');
ok(view.layout === undefined || view.layout === null || view._waiting,
   'no layout built while waiting');
await flush();
ok(view._waiting === false, 'clears once the sheet resolves');
ok(!!view.layout, 'layout built only after the tiles');

console.log('2. the sheet is fetched once for the page');
const before = atlasLoads;
await settle(fam('minimal')); await settle(fam('algebraic')); await settle(all);
ok(atlasLoads === before, `no refetch across 3 more subsets (loads=${atlasLoads})`);

console.log('3. cold vs warm');
view.cache.clear();                    // test 2 already cached `all`
const t1=await settle(all); const a1=snap();
await settle(fam('algebraic'));
const t3=await settle(all); const a3=snap();
console.log(`   cold=${t1}ms warm=${t3}ms`);
ok(same(a1,a3),'returning restores the identical layout');
ok(t3*4<t1||t3<5,`warm much faster (${t3}ms vs ${t1}ms)`);

console.log('4. every family there and back');
let mism=0,worstW=0; const first={};
for(const f of [...new Set(all.map(e=>e.primary_family))]){await settle(fam(f));first[f]=snap();}
for(const f of Object.keys(first)){worstW=Math.max(worstW,await settle(fam(f)));if(!same(first[f],snap()))mism++;}
ok(mism===0,`all 16 restore identically; worst warm ${worstW}ms`);

console.log('5. bounded, and a resize misses');
for(let i=0;i<60;i++) await settle(all.slice(0,5+i));
ok(view.cache.size<=32,`size ${view.cache.size} <= 32`);
await settle(fam('quadric'));
const k1=view._key; W=1200;H=900; await settle(fam('quadric')); const k2=view._key;
ok(k1!==k2,'a different canvas size is a different key');
W=840;H=740;
console.log(fail?`\nRESULT: ${fail} FAILURE(S)`:'\nRESULT: OK');
process.exit(fail?1:0);
