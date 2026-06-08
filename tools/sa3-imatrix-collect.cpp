// sa3-imatrix-collect: run DiT over calibration prompts, collect importance matrix -> file.
//   ./sa3-imatrix-collect <t5.gguf> <dit.gguf> <same.gguf> <out.imatrix>
#include "gguf-weights.h"
#include "sa3-tokenizer.h"
#include "sa3-t5gemma-enc.h"
#include "sa3-imatrix.h"
#include "sa3-dit.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

static const char * PROMPTS[] = {
    "8-bit retro arcade power-up jingle, chiptune, square wave",
    "laser zap retro arcade shoot sfx, bright and punchy",
    "deep cinematic boom impact, sub bass, trailer hit",
    "coin pickup blip, classic platformer, short",
    "ambient sci-fi drone, eerie pad, evolving texture",
    "explosion debris, gritty, layered noise burst",
    "magic spell sparkle, shimmering bells, fantasy ui",
    "heavy footstep on metal, mechanical clank, robot",
    "upbeat electronic loop, driving bassline, synth lead",
    "water splash droplet, organic, clean foley",
    "engine rev car, aggressive, motorsport",
    "menu select click, soft ui beep, modern",
};

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s <t5> <dit> <same> <out.imatrix>\n", argv[0]); return 1; }
    const char * t5p=argv[1], *ditp=argv[2], *samep=argv[3], *outp=argv[4];
    const int CD=768, C=256, MAXTOK=256, STEPS=8;

    GGUFModel cg; if (!gf_load(&cg, samep)) return 1;
    const float * sec_w=(const float*)gf_get_data(cg,"sec.emb.w");
    const float * sec_b=(const float*)gf_get_data(cg,"sec.emb.b");
    const float * pad_e=(const float*)gf_get_data(cg,"prompt.pad_embed");
    std::vector<float> secW(sec_w,sec_w+(size_t)256*768), secB(sec_b,sec_b+768), padE(pad_e,pad_e+768);
    gf_close(&cg);
    auto seconds_embed=[&](float s, std::vector<float>&sec){ sec.assign(CD,0);
        float norm=(s<0?0:s>384?384:s)/384.f; int half=128; float lo=logf(0.5f),hi=logf(10000.f);
        std::vector<float> ef(256);
        for(int j=0;j<half;j++){float f=expf((float)j/(half-1)*(hi-lo)+lo);float a=norm*f*2.f*(float)M_PI;ef[j]=cosf(a);ef[half+j]=sinf(a);}
        for(int o=0;o<CD;o++){double acc=secB[o];for(int i=0;i<256;i++)acc+=(double)secW[(size_t)o*256+i]*ef[i];sec[o]=(float)acc;} };

    struct ggml_context*meta=nullptr; struct gguf_init_params gp={true,&meta};
    struct gguf_context*tg=gguf_init_from_file(t5p,gp); SA3Tokenizer tok;
    if(!tg||!sa3tok_load_from_gguf(&tok,tg))return 1; gguf_free(tg); ggml_free(meta);

    SA3T5GModel t5; if(!sa3t5g_load(&t5,t5p))return 1;
    g_sa3_imx.enabled = true;          // turn on collection BEFORE DiT load/runs
    SA3DiT dit; if(!sa3dit_load(&dit,ditp))return 1;

    int NP=(int)(sizeof(PROMPTS)/sizeof(PROMPTS[0]));
    float secs_list[]={10,4,8,2,12,3,5,4,16,2,6,1};
    for(int p=0;p<NP;p++){
        std::vector<int> ids,valid; sa3tok_encode_padded(&tok,PROMPTS[p],MAXTOK,ids,valid);
        std::vector<float> t5out((size_t)CD*MAXTOK); sa3t5g_encode(&t5,ids.data(),valid.data(),MAXTOK,t5out.data());
        std::vector<float> sec; seconds_embed(secs_list[p],sec);
        int cross_T=MAXTOK+1; std::vector<float> cross((size_t)CD*cross_T);
        for(int t=0;t<MAXTOK;t++)for(int d=0;d<CD;d++)cross[(size_t)t*CD+d]=valid[t]?t5out[(size_t)t*CD+d]:padE[d];
        for(int d=0;d<CD;d++)cross[(size_t)MAXTOK*CD+d]=sec[d];
        float s=secs_list[p]; int ds=4096,align=8192; long tg2=(long)ceilf((s+6.f)*44100.f); tg2=((tg2+align-1)/align)*align; int L=(int)(tg2/ds);
        std::mt19937 rng((unsigned)p); std::normal_distribution<float> nd(0,1);
        std::vector<float> x((size_t)C*L); for(auto&v:x)v=nd(rng);
        std::vector<float> v((size_t)C*L);
        std::vector<float> sig(STEPS+1); for(int i=0;i<=STEPS;i++){float tl=1.f-(float)i/STEPS;sig[i]=i==0?1.f:i==STEPS?0.f:1.f/(1.f+expf(-(tl*8.2f-2.f)));}
        for(int i=0;i<STEPS;i++){ sa3dit_forward(&dit,x.data(),L,sig[i],cross.data(),cross_T,sec.data(),v.data()); float dt=sig[i+1]-sig[i]; for(size_t k=0;k<x.size();k++)x[k]+=dt*v[k]; }
        fprintf(stderr,"[collect] prompt %d/%d L=%d done\n",p+1,NP,L);
    }
    sa3dit_free(&dit); sa3t5g_free(&t5);
    sa3_imx_save(outp);
    return 0;
}
