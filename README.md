# Clearcut

A free background remover that runs entirely in the browser. Photos are never uploaded.

## How it works

1. **AI segmentation.** The `Best` mode uses IS-Net (DIS), a model built for complex scenes, hair and fine detail. It runs at 1024 × 1024. `Fast` mode uses U²-Net-p for simple photos.
2. **Graphics card when available.** It runs on WebGPU where the browser supports it. Otherwise it falls back to multi-threaded WebAssembly on the processor.
3. **Full-resolution refinement.** This runs in a background worker:
   - stray-patch removal
   - a guided filter that snaps the mask to real edges in the photo
   - foreground colour estimation, which removes background colour bleeding into hair and fur
4. **Manual touch-up.** Erase and Restore brushes, plus Cutoff and Softness sliders.

The model is downloaded on first use and saved in the browser's Cache Storage, so later visits load it instantly.

## Files

| Path | What it is |
| --- | --- |
| `index.html` | The page and the editor |
| `worker.js` | Model loading, inference and edge refinement |
| `ort/` | ONNX Runtime Web 1.20.1, self-hosted (the engine is split into `engine.part0–2.bin`) |
| `models/isnet.part0–4.bin` | IS-Net with 8-bit weights, split into ~10 MB pieces (GitHub web uploads allow 25 MB per file) |
| `models/u2netp.onnx` | Fast model |
| `vercel.json` | Headers that enable multi-threading, plus long caching for the model files |

The total size is about 70 MB, under Vercel's 100 MB upload limit on the free Hobby plan.

## Deploy

**GitHub + Vercel:** push this folder to a repo, then on vercel.com go to Add New → Project, import the repo, set the Framework Preset to "Other" and click Deploy.

**CLI:** run `npx vercel --prod` inside this folder.

The page must be served over http(s). Opening `index.html` as a file won't work.

## Local preview

```
npx serve .
```
Multi-threading only turns on when the `vercel.json` headers are present, so a local preview runs single-threaded.

## Credits

- IS-Net / DIS, Xuebin Qin et al., Apache-2.0: https://github.com/xuebinqin/DIS
- U²-Net, Xuebin Qin et al., Apache-2.0: https://github.com/xuebinqin/U-2-Net
- ONNX Runtime Web, MIT: https://onnxruntime.ai
- Foreground estimation follows "Approximate Fast Foreground Colour Estimation" (Forte & Pitié, ICIP 2021).
