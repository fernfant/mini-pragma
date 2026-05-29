/* Shared forward-pass flow engine.
   Usage:  initFlow(containerEl, {focus:'all'|'encode'|'attention'|'neurons'|'output', preset:0})
   The container gets the .fp class, a canvas, controls, and a caption built in.
   One real (tiny, trained) network drives every instance, so predictions are honest. */
(function(){
  const D=6,V=5,H1=10;
  const vocab=["time","blue","you","much","cake"];
  const presets=[
    {tokens:["once","upon","a"],        E:[[.6,-.2,.3,.1,-.5,.4],[-.3,.5,-.6,.2,.7,-.1],[.2,.8,.1,-.4,-.2,.5]]},
    {tokens:["the","sky","is"],         E:[[.8,-.3,.5,-.6,.2,.1],[-.4,.9,-.2,.7,-.5,.6],[.3,.2,.9,-.1,.8,-.4]]},
    {tokens:["happy","birthday","to"],  E:[[.5,.6,-.4,.2,-.7,.3],[.7,-.2,.6,-.5,.4,.1],[-.3,.8,.2,.6,-.1,-.6]]},
    {tokens:["thank","you","very"],     E:[[-.6,.4,.7,-.2,.5,-.3],[.2,-.5,.4,.8,-.6,.3],[.6,.3,-.4,-.1,.7,.5]]},
  ];
  const W1=[[-0.111,0.758,0.285,-0.237,-0.565,-0.388,0.544,0.114,0.91,0.466],[1.96,0.05,-0.793,1.321,0.801,0.027,-0.873,-0.915,-1.279,0.529],[-0.894,1.136,1.459,-0.633,0.211,-1.138,-0.329,0.696,0.444,2.146],[-1.913,-1.502,-0.667,0.662,1.244,0.869,-0.218,-0.442,-0.062,1.074],[-2.661,0.372,0.388,-1.492,-0.446,1.16,-1.004,-0.267,1.909,-0.915],[1.696,-0.051,-1.375,-0.06,-0.658,1.563,0.701,0.198,1.036,-2.713]];
  const b1=[0.473,0.184,0.213,0.889,0.447,0.648,-0.064,-0.282,1.59,0.643];
  const W2=[[3.351,-0.972,-0.394,-2.403,-0.923],[-0.161,1.601,-1.621,-0.934,-0.132],[0.482,1.42,-1.044,-1.932,0.055],[0.831,-1.501,1.559,-0.355,-0.344],[-0.122,-0.245,1.893,0.244,-0.065],[0.523,-1.558,-0.211,2.065,0.054],[-0.985,-0.317,0.421,-0.906,-0.092],[0.49,-0.592,0.791,0.263,-0.091],[-0.246,0.82,-1.196,2.535,-1.123],[-2.042,1.527,2.32,-1.522,0.076]];
  const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
  const softmax=a=>{const m=Math.max(...a),e=a.map(x=>Math.exp(x-m)),s=e.reduce((x,y)=>x+y,0);return e.map(x=>x/s);};

  window.initFlow=function(cont, opts){
    opts=opts||{}; const focus=opts.focus||'all';
    const showSliders = (focus==='all'||focus==='attention');
    cont.classList.add('fp'); cont.setAttribute('tabindex','0');
    if(!cont.getAttribute('aria-label')) cont.setAttribute('aria-label','Interactive forward-pass animation. Space plays or pauses, right arrow steps through stages, R restarts.');
    cont.innerHTML =
      '<canvas class="fp-canvas" width="1180" height="560" role="img" aria-label="A forward pass: words are encoded into number vectors; attention beams blend them weighted by relevance; a layer of neurons fires; an output distribution over candidate next-words resolves to a single winning prediction."></canvas>'+
      '<div class="fp-panel">'+
        '<div class="fp-group"><span class="fp-lbl">Sentence</span><div class="fp-row fp-sentences"></div></div>'+
        '<div class="fp-group"><span class="fp-lbl">Run</span><div class="fp-row">'+
          '<button class="fp-play">▶ Play</button><button class="fp-step">⏭ Step</button><button class="fp-restart">↺ Restart</button></div></div>'+
        (showSliders?('<div class="fp-group fp-sliders"><span class="fp-lbl">Attention — you set the blend <button class="fp-auto">auto</button></span>'+
          '<div class="fp-s"><span class="fp-t0n">the</span><input type="range" class="fp-a0" min="0" max="100" value="33" aria-label="attention weight on word 1"><b class="fp-a0v">33%</b></div>'+
          '<div class="fp-s"><span class="fp-t1n">cat</span><input type="range" class="fp-a1" min="0" max="100" value="17" aria-label="attention weight on word 2"><b class="fp-a1v">17%</b></div>'+
          '<div class="fp-s"><span class="fp-t2n">sat</span><input type="range" class="fp-a2" min="0" max="100" value="49" aria-label="attention weight on word 3"><b class="fp-a2v">49%</b></div></div>'):'')+
        '<div class="fp-legend">'+
          '<span><i class="fp-dot" style="background:var(--fp-cool)"></i>encode</span>'+
          '<span><i class="fp-dot" style="background:var(--fp-warm)"></i>attention</span>'+
          '<span><i class="fp-dot" style="background:var(--fp-neu)"></i>neurons fire</span>'+
          '<span><i class="fp-dot" style="background:var(--fp-good)"></i>prediction</span></div>'+
      '</div>'+
      '<p class="fp-cap">'+(opts.caption||'Every number is real — the network was trained so each phrase predicts its true next word. Focus the panel, then <kbd>Space</kbd> play/pause · <kbd>&rarr;</kbd> step · <kbd>R</kbd> restart.')+'</p>';

    const cv=cont.querySelector('.fp-canvas'), ctx=cv.getContext('2d');
    const W=cv.width,H=cv.height, dpr=Math.min(window.devicePixelRatio||1,2);
    cv.width=W*dpr; cv.height=H*dpr; ctx.scale(dpr,dpr); cv.style.aspectRatio=W+'/'+H;

    let pi=opts.preset||0, comp={}, attnAuto=true, attnUser=[33,17,49];
    function recompute(){
      const P=presets[pi], E=P.E, fc=2;
      const scores=E.map(ej=>dot(E[fc],ej)/Math.sqrt(D));
      const auto=softmax(scores);
      let attn;
      if(attnAuto){ attn=auto; attnUser=auto.map(x=>Math.round(x*100)); syncSliders(); }
      else { const s=attnUser.reduce((a,b)=>a+b,0)||1; attn=attnUser.map(x=>x/s); }
      const blend=Array.from({length:D},(_,d)=>E.reduce((s,ej,j)=>s+attn[j]*ej[d],0));
      const pre=Array.from({length:H1},(_,h)=>blend.reduce((s,x,d)=>s+x*W1[d][h],0)+b1[h]);
      const hid=pre.map(x=>Math.max(0,x));
      const logits=Array.from({length:V},(_,v)=>hid.reduce((s,x,h)=>s+x*W2[h][v],0));
      const probs=softmax(logits);
      comp={E,attn,auto,blend,hid,logits,probs,winner:probs.indexOf(Math.max(...probs)),tokens:P.tokens,cand:vocab,focus:fc};
    }

    const lane=[150,280,410], colX={tok:90,emb:300,att:545,mlp:800,out:1045};
    const cellH=13,cellW=22, stageX=[colX.emb,colX.att,colX.mlp,colX.out,colX.out+30];
    const caps=[
      'Each word becomes a <b>vector of numbers</b> — its meaning written as coordinates.',
      'The focus word <b>gathers meaning</b> from every word, weighted by relevance.',
      'A layer of <b>neurons fires</b>; ReLU silences the rest. This sparse pattern is the "thinking".',
      'The firing pattern becomes <b>scores over candidate words</b>, squashed by softmax to add to 100%.',
      'The model <b>commits</b>: the brightest word wins.'
    ];
    const capEl=cont.querySelector('.fp-cap'), defaultCap=capEl.innerHTML;
    // dim non-focused stages
    function fdim(stage){ return (focus==='all'||focus===stage)?1:0.16; }

    function vcol(v,a){a=a==null?1:a;const t=Math.max(-1,Math.min(1,v));let R,G,B;
      if(t>=0){R=74+(255-74)*t;G=163+(106-163)*t;B=255+(43-255)*t;}
      else{const u=-t;R=74+(20-74)*u;G=163+(120-163)*u;B=255+(80-255)*u;}
      return`rgba(${R|0},${G|0},${B|0},${a})`;}

    let mode='play', playing=true, speed=1, stepIdx=-1, T=0;
    let t0=performance.now(), pausedPhase=0, stepT0=performance.now();
    const PXPS=185, SWEEP=(W+130)/PXPS, BLOOM=2.0, CYCLE=SWEEP+BLOOM;
    let wf=-40, bloomT=0, raf=0;
    const motes=Array.from({length:30},()=>({x:Math.random()*W,y:80+Math.random()*(H-160),s:.2+Math.random()*.5,r:.6+Math.random()*1.4}));

    // when focused on one stage, start fully revealed (others dimmed) so the stage sits in context
    if(focus!=='all'){ mode='step'; stepIdx=4; }

    function reveal(x){return Math.max(0,Math.min(1,(wf-x+30)/95));}
    function phaseNow(){const p=playing?((performance.now()-t0)*speed/1000):pausedPhase;return((p%CYCLE)+CYCLE)%CYCLE;}
    function updateState(){
      if(mode==='step'){wf=stepIdx<0?-40:stageX[Math.min(stepIdx,stageX.length-1)];
        bloomT=stepIdx>=3?Math.min(1,(performance.now()-stepT0)/1200):0;return;}
      const ph=phaseNow();
      if(ph<SWEEP){wf=-40+(ph/SWEEP)*(W+130);bloomT=0;}
      else{wf=W+90;bloomT=Math.min(1,(ph-SWEEP)/BLOOM);}
    }

    function draw(ts){
      updateState();
      ctx.clearRect(0,0,W,H);
      drawMotes(ts);
      ctx.save();ctx.globalAlpha=.55;ctx.fillStyle='#5c5e6e';ctx.font='11px -apple-system,sans-serif';ctx.textAlign='center';
      [['encode',colX.emb,'encode'],['attention',colX.att,'attention'],['neurons',colX.mlp,'neurons'],['output',colX.out,'output']].forEach(([l,x,st])=>{ctx.globalAlpha=.55*fdim(st);ctx.fillText(l.toUpperCase(),x,34);});
      ctx.restore();
      edgesTokEmb(ts); attention(ts); embToNeurons(ts); output(ts); tokensDraw(); embeddings(ts);
      const sweepDone=mode==='play'?wf>=colX.out:stepIdx>=3;
      if(!sweepDone&&wf>colX.tok&&wf<colX.out-100){const fade=Math.min(1,(colX.out-100-wf)/130);
        ctx.save();ctx.globalAlpha=.55*fade;ctx.strokeStyle='rgba(255,170,110,.9)';ctx.lineWidth=2;
        ctx.shadowBlur=16;ctx.shadowColor='rgba(255,140,80,.9)';
        ctx.beginPath();ctx.moveTo(wf,72);ctx.lineTo(wf,H-60);ctx.stroke();ctx.restore();}
    }
    function drawMotes(ts){ctx.save();
      motes.forEach(m=>{m.x+=m.s*0.15;if(m.x>W+5)m.x=-5;
        ctx.globalAlpha=.06+.05*Math.sin(ts/1400+m.x);ctx.fillStyle='#9fb0ff';
        ctx.beginPath();ctx.arc(m.x,m.y,m.r,0,7);ctx.fill();});ctx.restore();}
    function tokensDraw(){ctx.textAlign='center';ctx.textBaseline='middle';
      comp.tokens.forEach((w,i)=>{const a=reveal(colX.tok)*fdim('encode');const slide=(1-reveal(colX.tok))*-26;
        ctx.save();ctx.globalAlpha=a;
        ctx.fillStyle='#171a26';ctx.strokeStyle=i===comp.focus?'rgba(255,106,43,.85)':'#2c2f3d';ctx.lineWidth=1.5;
        rr(colX.tok-42+slide,lane[i]-18,84,36,9);ctx.fill();ctx.stroke();
        ctx.fillStyle=i===comp.focus?'#ffd9c7':'#ece9e3';ctx.font='15px -apple-system,sans-serif';
        ctx.fillText('"'+w+'"',colX.tok+slide,lane[i]);ctx.restore();});}
    function embeddings(ts){comp.E.forEach((vec,i)=>{const a=reveal(colX.emb)*fdim('encode');const y0=lane[i]-(D*cellH)/2;
      vec.forEach((val,d)=>{const rise=Math.max(0,Math.min(1,(wf-(colX.emb-20))/60));
        ctx.save();ctx.globalAlpha=a;ctx.fillStyle=vcol(val,.35+.6*Math.min(1,Math.abs(val)));
        ctx.shadowBlur=9*rise;ctx.shadowColor=vcol(val,.6);
        rr(colX.emb-cellW/2,y0+d*cellH,cellW,cellH-2,3);ctx.fill();ctx.restore();});});}
    function edgesTokEmb(ts){comp.tokens.forEach((w,i)=>{const a=reveal(colX.emb-40)*.5*fdim('encode');if(a<=0)return;
      ctx.save();ctx.globalAlpha=a;ctx.strokeStyle='rgba(74,163,255,.5)';ctx.lineWidth=1;
      ln(colX.tok+44,lane[i],colX.emb-cellW/2-4,lane[i]);
      flowParticles(colX.tok+44,lane[i],colX.emb-cellW/2-4,lane[i],ts,a,i);ctx.restore();});}
    function attention(ts){const bx=colX.att,by=lane[comp.focus],a=reveal(colX.att)*fdim('attention');if(a<=0)return;
      comp.attn.forEach((w,j)=>{const sx=colX.emb+cellW/2+2,sy=lane[j];
        ctx.save();ctx.globalAlpha=a*Math.min(1,.25+w);
        ctx.strokeStyle=`rgba(255,106,43,${.18+.7*w})`;ctx.lineWidth=1+11*w;
        ctx.shadowBlur=14*w;ctx.shadowColor='rgba(255,106,43,.6)';
        cv2(sx,sy,bx-30,by);ctx.stroke();ctx.restore();
        flowParticlesBez(sx,sy,(sx+bx)/2,sy,(sx+bx)/2,by,bx-30,by,ts,w,a,j);
        ctx.save();ctx.globalAlpha=a;ctx.fillStyle='rgba(255,170,120,'+a+')';ctx.font='11px sans-serif';ctx.textAlign='center';
        ctx.fillText((w*100|0)+'%',(sx+bx)/2,sy+(j<comp.focus?-9:15));ctx.restore();});
      const y0=by-(D*cellH)/2;
      comp.blend.forEach((val,d)=>{ctx.save();ctx.globalAlpha=a;
        ctx.fillStyle=vcol(val,.4+.6*Math.min(1,Math.abs(val)));ctx.shadowBlur=12;ctx.shadowColor=vcol(val,.7);
        rr(bx-cellW/2,y0+d*cellH,cellW,cellH-2,3);ctx.fill();ctx.restore();});
      ctx.save();ctx.globalAlpha=a;ctx.fillStyle='#8b8c98';ctx.font='11px sans-serif';ctx.textAlign='center';
      ctx.fillText('blend',bx,y0-12);ctx.restore();}
    function embToNeurons(ts){const bx=colX.att,by=lane[comp.focus],mx=colX.mlp,a=reveal(colX.mlp)*fdim('neurons');if(a<=0)return;
      const ny0=130,nstep=360/(H1-1);
      for(let h=0;h<H1;h++){const act=comp.hid[h],na=Math.min(1,act/2.2);
        ctx.save();ctx.globalAlpha=a*(.08+.5*na);
        ctx.strokeStyle=`rgba(201,160,255,${.1+.6*na})`;ctx.lineWidth=.6+2.4*na;
        cv2(bx+cellW/2,by,mx-14,ny0+h*nstep);ctx.stroke();
        if(na>0.05) flowParticlesBez(bx+cellW/2,by,(bx+mx)/2,by,(bx+mx)/2,ny0+h*nstep,mx-14,ny0+h*nstep,ts,na*.5,a,h+3,'rgba(210,180,255,');
        ctx.restore();}
      for(let h=0;h<H1;h++){const act=comp.hid[h],na=Math.min(1,act/2.2),fired=act>0;
        const pulse=fired?(.6+.4*Math.sin(ts*speed/240+h)):.15;
        const rev=Math.max(0,Math.min(1,(wf-(mx-20)-h*5)/45));
        ctx.save();ctx.globalAlpha=a*Math.max(.2,rev);
        ctx.fillStyle=fired?`rgba(201,160,255,${.25+.7*na*pulse})`:'rgba(70,72,86,.5)';
        ctx.shadowBlur=fired?18*na*pulse:0;ctx.shadowColor='rgba(201,160,255,.9)';
        ctx.beginPath();ctx.arc(mx,ny0+h*nstep,(7+6*na)*(.6+.4*rev),0,7);ctx.fill();ctx.restore();}
      ctx.save();ctx.globalAlpha=a;ctx.fillStyle='#8b8c98';ctx.font='11px sans-serif';ctx.textAlign='center';
      ctx.fillText('ReLU layer',mx,ny0-18);ctx.restore();}
    function output(ts){const mx=colX.mlp,ox=colX.out,a=reveal(ox)*fdim('output');if(a<=0)return;
      const ny0=130,nstep=360/(H1-1),by0=120,barH=64,gap=14;
      for(let h=0;h<H1;h++){if(comp.hid[h]<=0)continue;
        for(let v=0;v<V;v++){const st=comp.probs[v];
          ctx.save();ctx.globalAlpha=a*.07*(.4+st);ctx.strokeStyle='rgba(70,211,154,.5)';ctx.lineWidth=.5;
          ln(mx+8,ny0+h*nstep,ox-90,by0+v*(barH+gap)+barH/2);ctx.stroke();ctx.restore();}}
      for(let v=0;v<V;v++){const y=by0+v*(barH+gap),isWin=v===comp.winner;
        const grow=isWin?a*(.55+.45*bloomT):a;const len=20+260*comp.probs[v]*grow;
        ctx.save();ctx.globalAlpha=a;
        ctx.fillStyle='#171a26';rr(ox-90,y+barH/2-9,300,18,9);ctx.fill();
        ctx.fillStyle=isWin?'rgba(70,211,154,.9)':'rgba(120,124,140,.7)';
        if(isWin){ctx.shadowBlur=24*bloomT+6;ctx.shadowColor='rgba(70,211,154,.9)';}
        rr(ox-90,y+barH/2-9,len,18,9);ctx.fill();ctx.shadowBlur=0;
        ctx.fillStyle=isWin?'#bff5dd':'#a9abb8';ctx.font=(isWin?'600 ':'')+(14+(isWin?2*bloomT:0))+'px -apple-system,sans-serif';
        ctx.textAlign='left';ctx.textBaseline='middle';ctx.fillText('"'+comp.cand[v]+'"',ox-90+8,y+barH/2);
        ctx.textAlign='right';ctx.fillStyle=isWin?'#bff5dd':'#7c7e8c';
        ctx.fillText((comp.probs[v]*100).toFixed(0)+'%',ox+208,y+barH/2);ctx.restore();}
      if(bloomT>0){ctx.save();ctx.globalAlpha=bloomT;ctx.textAlign='center';
        ctx.fillStyle='rgba(70,211,154,'+bloomT+')';ctx.font='600 16px -apple-system,sans-serif';
        ctx.fillText('predicted next word',ox+60,by0-26);ctx.restore();}}

    function flowParticles(x1,y1,x2,y2,ts,a,seed){const n=2;
      for(let k=0;k<n;k++){const p=((ts*speed/1100)+seed*.4+k/n)%1;const x=x1+(x2-x1)*p,y=y1+(y2-y1)*p;
        ctx.beginPath();ctx.fillStyle='rgba(120,180,255,'+(a*.7)+')';ctx.arc(x,y,1.6,0,7);ctx.fill();}}
    function flowParticlesBez(x1,y1,cx1,cy1,cx2,cy2,x2,y2,ts,w,a,seed,col){
      col=col||'rgba(255,200,150,';const n=Math.max(1,Math.round(1+w*4));
      for(let k=0;k<n;k++){const p=((ts*speed/950)+seed*.33+k/n)%1;const pt=bez(x1,y1,cx1,cy1,cx2,cy2,x2,y2,p);
        ctx.beginPath();ctx.fillStyle=col+(a*(.4+.6*w))+')';ctx.arc(pt.x,pt.y,1.5+3*w,0,7);ctx.fill();}}
    function rr(x,y,w,h,r){ctx.beginPath();ctx.moveTo(x+r,y);ctx.arcTo(x+w,y,x+w,y+h,r);ctx.arcTo(x+w,y+h,x,y+h,r);ctx.arcTo(x,y+h,x,y,r);ctx.arcTo(x,y,x+w,y,r);ctx.closePath();}
    function ln(a,b,c,d){ctx.beginPath();ctx.moveTo(a,b);ctx.lineTo(c,d);ctx.stroke();}
    function cv2(x1,y1,x2,y2){const m=(x1+x2)/2;ctx.beginPath();ctx.moveTo(x1,y1);ctx.bezierCurveTo(m,y1,m,y2,x2,y2);}
    function bez(x1,y1,cx1,cy1,cx2,cy2,x2,y2,t){const u=1-t;
      return{x:u*u*u*x1+3*u*u*t*cx1+3*u*t*t*cx2+t*t*t*x2,y:u*u*u*y1+3*u*u*t*cy1+3*u*t*t*cy2+t*t*t*y2};}

    const playBtn=cont.querySelector('.fp-play');
    function loopStart(){if(raf)return;(function go(ts){T=ts||performance.now();draw(T);raf=requestAnimationFrame(go);})();}
    function setCap(){capEl.innerHTML=(mode==='step'&&stepIdx>=0&&focus==='all')?caps[Math.min(stepIdx,caps.length-1)]:defaultCap;}
    function play(){if(mode==='step'){mode='play';t0=performance.now();}else{t0=performance.now()-pausedPhase*1000/speed;}
      playing=true;playBtn.textContent='⏸ Pause';setCap();}
    function pause(){pausedPhase=phaseNow();playing=false;playBtn.textContent='▶ Play';}
    function restart(){mode='play';t0=performance.now();pausedPhase=0;bloomT=0;stepIdx=-1;playing=true;playBtn.textContent='⏸ Pause';setCap();}
    function stepFwd(){mode='step';playing=false;playBtn.textContent='▶ Play';stepIdx=Math.min(stepIdx+1,stageX.length-1);stepT0=performance.now();setCap();}
    playBtn.addEventListener('click',()=>playing?pause():play());
    cont.querySelector('.fp-step').addEventListener('click',stepFwd);
    cont.querySelector('.fp-restart').addEventListener('click',restart);
    cont.addEventListener('keydown',e=>{
      if(e.code==='Space'){e.preventDefault();playing?pause():play();}
      else if(e.key==='ArrowRight'){e.preventDefault();stepFwd();}
      else if(e.key==='r'||e.key==='R'){restart();}});

    const sc=cont.querySelector('.fp-sentences');
    presets.forEach((p,i)=>{const b=document.createElement('button');b.className='chip';
      b.textContent=p.tokens.join(' ');b.setAttribute('aria-pressed',i===pi);
      b.onclick=()=>{pi=i;attnAuto=true;[...sc.children].forEach((c,k)=>c.setAttribute('aria-pressed',k===i));
        recompute();labelSliders();
        if(focus==='all'){restart();} else {mode='step';stepIdx=4;stepT0=performance.now();playing=false;playBtn.textContent='▶ Play';}};
      sc.appendChild(b);});

    let aEls=[],aVs=[];
    function syncSliders(){if(!showSliders)return;aEls.forEach((el,i)=>{el.value=attnUser[i];aVs[i].textContent=Math.round(comp.attn?comp.attn[i]*100:attnUser[i])+'%';});}
    function labelSliders(){if(!showSliders)return;const t=presets[pi].tokens;
      cont.querySelector('.fp-t0n').textContent=t[0];cont.querySelector('.fp-t1n').textContent=t[1];cont.querySelector('.fp-t2n').textContent=t[2];}
    if(showSliders){
      aEls=[0,1,2].map(i=>cont.querySelector('.fp-a'+i));
      aVs=[0,1,2].map(i=>cont.querySelector('.fp-a'+i+'v'));
      aEls.forEach((el,i)=>el.addEventListener('input',()=>{attnAuto=false;attnUser[i]=+el.value;recompute();
        const s=attnUser.reduce((a,b)=>a+b,0)||1;aVs.forEach((v,k)=>v.textContent=Math.round(attnUser[k]/s*100)+'%');}));
      cont.querySelector('.fp-auto').addEventListener('click',()=>{attnAuto=true;recompute();});
    }

    recompute();labelSliders();syncSliders();loopStart();
    if(focus==='all') play(); else { playBtn.textContent='▶ Play'; }
  };
})();
