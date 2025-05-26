rem you should change the root with your own environment path root.
rem and you could change the ENV_NAME with your one vitual environment.
set root=C:\Users\Seo\anaconda3
set ENV_NAME=DiffV2IR

if not exist "%root%" (
    echo check the root please: %root%
    pause
    exit
)

call %root%\Scripts\activate.bat %root%

echo enter the virtual environment.
call conda activate %ENV_NAME%

echo start downloading environment for %ENV_NAME%.
call pip install imageio imageio-ffmpeg pytorch-lightning omegaconf test-tube streamlit einops ^
 torch-fidelity transformers torchmetrics kornia wandb openai gradio seaborn ^
 git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers ^
 git+https://github.com/openai/CLIP.git@main#egg=clip ^
 git+https://github.com/crowsonkb/k-diffusion.git

call conda deactivate

echo complete. 