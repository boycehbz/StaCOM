
import sys
sys.path.append('./')
from utils.video_processing import *

root = 'output/diffusion_flow_train/08.22-12h46m28s/images'
output = 'output/vis_results/train.mp4'

single_video = True
FPS = 30

if single_video:
    src = root
    dst = output
    os.makedirs(os.path.dirname(output), exist_ok=True)
    generate_mp4(src, dst, output_fps=FPS)

else:
    files = os.listdir(root)
    for f in files:
        name = f + '.mp4'
        src = os.path.join(root, f)
        dst = os.path.join(output, name)

        generate_mp4(src, dst, output_fps=FPS)

# for f in files:
#     name = f.split('.')[0]
#     src = os.path.join(root, f)
#     dst = os.path.join(output, name)

#     os.makedirs(dst, exist_ok=True)

#     cut_video(src, dst, 1, 1)




