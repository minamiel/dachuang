# Text Enhancement Workflow

This repo now supports two parallel tracks for text enhancement:

- A stable baseline using Real-ESRGAN
- A diffusion experiment branch for crop-only text restoration

## Recommended evaluation flow

1. Put real text crops into `eval_inputs/`
2. Run the unified evaluation script
3. Compare results under `eval_outputs/comparisons/`

If your source images are full phone photos, crop text regions first:

```bash
python tools/crop_eval_inputs.py -i path/to/phone_photos -o eval_inputs
```

The crop tool saves images like `eval_001.png` and writes crop metadata to `eval_inputs/manifest.csv`.

Example:

```bash
python tools/evaluate_text_models.py --input_dir eval_inputs --output_dir eval_outputs --methods bicubic,realesrgan,diffusion --realesrgan_model RealESRGAN_x4plus --diffusion_model_path experiments/diffusion_textzoom_bs8_latest.pth
```

## Recommended baseline

For the first usable demo, prefer Real-ESRGAN on cropped text regions:

```bash
python inference_realesrgan.py -i eval_inputs -o eval_outputs/realesrgan -n RealESRGAN_x4plus
```

## Recommended diffusion training

Train on text crops, not full photos. Use real text crops first, then add synthetic data only as a supplement.

Example:

```bash
python train_diffusion.py --cond_mode concat --batch_size 8 --epochs 200 --hr_size 256 --train_size 128 --lr 1e-4 --lambda_seg 0 --num_workers 4 --hr_dir dataset/HR --save_dir experiments --experiment_name diffusion_textzoom_bs8
```

Resume:

```bash
python train_diffusion.py --cond_mode concat --batch_size 8 --epochs 200 --hr_size 256 --train_size 128 --lr 1e-4 --lambda_seg 0 --num_workers 4 --hr_dir dataset/HR --save_dir experiments --experiment_name diffusion_textzoom_bs8 --resume
```

## Diffusion inference

Run diffusion only on cropped text regions:

```bash
python inference_diffusion.py -i eval_inputs -o eval_outputs/diffusion --model_path experiments/diffusion_textzoom_bs8_latest.pth --timesteps 1000
```
