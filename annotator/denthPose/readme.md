### 安装 detectron2
pip install 'git+https://github.com/facebookresearch/detectron2.git'

### 安装 DensePose
pip install git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose


#
python apply_net.py show densepose_rcnn_R_50_FPN_s1x.yaml model_final_162be9.pkl demo.jpg dp_contour,bbox --output image_densepose_contour.png

