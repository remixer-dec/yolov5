This repository represents a fork of YOLOv5 model (by Ultralytics) ported to Pytorch 1.3. Use at your own risk.  
The port was made to run the model on GTX 780 GPU (Pytorch 1.3 is the last build that supports this GPU).  
For all the information about YOLOv5, please refer to the [original repo](https://github.com/ultralytics/yolov5).  
  
### Requirements
- This version has requirement auto-update disabled, so make sure to manually install all the original requirements  
- For training you'll also need an apex module, you can install it using ```pip install pytorch-extension```  
- You might need to run ```pip install pillow<7``` if you face an error about PILLOW_VERSION not existing in PIL  

### Model format
- Pretrained models published by Ultralytics use a new compressed format. You can import these models in an environment with new versions of pytorch and export them in old format. See convert_model.ipynb for the code.  

### Support
- This port was made once, tested successfully (with GTX780/CUDA10.2), and won't be supported in the future (Pull requests are welcome!). If something goes wrong, try replacing files causing the errors with ones from the original repo.  
  
### Changes 
- Missing functions have been added to oldpt_polyfills.py  
- Files utils/general.py and utils/metrics.py have been changed to fix appearing type errors (to always perform conflicting fp16/fp32 operations in fp32).  
- Training code parts, relying on torch.cuda.amp have been replaced with a previous version, that was only relying on apex. That code doesn't seem to work without apex, so make sure that it's installed.  
- Default batch_size has been reduced to fit in GPUs with low amounts of memory.  

### Results
- Only small pretrained model released with v4 tag was tested, and the results are identical to the original.  