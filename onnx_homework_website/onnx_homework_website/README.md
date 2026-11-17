# ModelLab — PyTorch → ONNX → GitHub Pages

This is a static website with two working ONNX Runtime Web inference forms:

- **Regression:** 8 Energy Efficiency features → 1 heating-load prediction.
- **Classification:** 10 Indian Liver Patient features → 2 output logits → class 0/1.

## 1. Add your two trained model files

Your exported model files are **NOT INCLUDED in this ZIP** because they remain on your own PC. Copy these exact files from `C:\Users\prane\Desktop\class and reg\` (or wherever your Python scripts produced them):

```
onnx_homework_website/
├── index.html
├── styles.css
├── app.js
└── models/
    ├── energy_best_model.onnx
    └── indian_liver_best.onnx
```

These must be the **ONNX** models, not `.pth` PyTorch weights. Names must match exactly, including lowercase/uppercase.

Note: the medical classifier was developed for coursework and is not a diagnostic tool.

## 2. Run locally in VS Code

Open a terminal **inside the `onnx_homework_website` folder** and run:

```
python -m http.server 8000
```

Then open http://localhost:8000 . Do **not** double-click `index.html` using a `file://` URL; browser security rules prevent fetching model files that way.

The first page load requires internet access to retrieve `onnxruntime-web` from jsDelivr CDN. Subsequent runtime availability depends on browser caching.

## 3. Publish using GitHub Pages

1. Create a public GitHub repository, e.g. `ml-onnx-homework`.
2. Upload **all** website files **including the two `.onnx` files** within the `models` folder, maintaining folder structure.
3. Repository → **Settings** → **Pages** → Build and deployment → **Deploy from a branch**.
4. Choose branch `main` and folder `/ (root)`; click **Save**.
5. Visit the Pages URL GitHub displays, typically `https://YOUR_USERNAME.github.io/ml-onnx-homework/`.
6. Open both tabs, submit their sample inputs, and verify numerical predictions. Also test on a phone if desired.

If a model file exceeds GitHub's usual single-file upload or repository limitations, use Git LFS or a compatible static host as appropriate; ordinary small ONNX files work directly.

## 4. Important technical details

- **No Python needed on the website.** ONNX Runtime Web (WASM) runs predictions locally in the visitor's browser.
- `X1..X8` order for Energy Efficiency: relative compactness, surface area, wall area, roof area, overall height, orientation, glazing area, glazing area distribution.
- Liver input order: Age, Gender (Male=0/Female=1), TB, DB, Alkphos, Sgpt, Sgot, TP, ALB, A/G Ratio.
- Normalization is **already inside** the exported networks, so JavaScript sends raw input numbers. Do not standardize them again.
- Liver outputs are **logits**, and the JavaScript selects argmax. They are not calibrated medical probabilities.
- The 0.9548 regression test R² and error values are from your supplied training output; they do not recompute during browser use.
- Both ONNX files must be the original working exports from `regression.py` and `class.py`.
- GitHub Pages requires no backend, but the ONNX Runtime library is fetched from a CDN, requiring internet access.
- The user enters inputs, which are processed inside the browser. This project does not transmit those input values to a custom prediction server.

## 5. Troubleshooting

**"Missing ... .onnx"**: File not uploaded, wrong name, or wrong directory. Check `models/` in the GitHub repository.

**"ONNX runtime unavailable"**: Network/CDN blocked. Check the browser console and retry with internet connection.

**"Could not fetch ... use local server"**: Run `python -m http.server 8000` in the site folder, not double-clicking the HTML file.

**"ONNX loading failed"**: Verify you uploaded the actual exported ONNX file, not a `.pth` file or an HTML download error page.

## Homework note

Your classification dataset is imbalanced and its published test accuracy in your run was modest. Do not present the classifier as a clinically validated system. The purpose of the webpage is to demonstrate a successful ONNX deployment.
