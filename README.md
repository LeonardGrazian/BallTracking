# What is this repo?
This repo uses a Mask R-CNN model to highlight the position of the ball in amateur sports video. 
Amateur sports videos are often filmed on handheld user devices that are not optimized to capture small, fast moving balls. 
This makes it harder for the viewer to follow the action, and ultimately reduces their enjoyment of the game. 
By highlighting the ball, this code aims to improve the amateur sports viewer's experience so that they can focus on supporting their friends, family, or team

| Original  | Highlighted |
| ------------- | ------------- |
| <img width="400" alt="Original" src="https://user-images.githubusercontent.com/13444386/213975382-deb5d163-750f-4576-a378-fd947d7e1324.gif">  | <img width="400" alt="Annotated" src="https://user-images.githubusercontent.com/13444386/213975341-a2fca484-8064-4be3-9a34-8fb616d0c1b9.gif">  |

I've included 3 example videos from the [Sports Videos in the Wild (SVW)](http://cvlab.cse.msu.edu/project-svw.html) dataset. The original videos are in the `data/` directory, and the model results are stored in the `output/` directory. You can run the model on your own sports videos, see the next section

# How to run 
This code can be run entirely on the CPU or on the GPU with a CUDA enabled machine. If your local machine does not have CUDA, I recommend spinning up an EC2 instance with a NVIDIA GPU-Optimized AMI. You can install the required CUDA libraries with the below commands:
```
> sudo apt-get update 
> sudo apt-get install nvidia-cuda-toolkit
```
The below commands will highlight the ball for all videos in the `<input-dir>` directory and store the new videos in `<output-dir>`. 
If `<input-dir>` or `<output-dir>` is not specified, they will default to `data/` and `output/` respectively. 
It expects all files in  `<input-dir>` to be MP4's. 
The commands are the same whether you're running on the CPU or GPU 
```
> git clone https://github.com/LeonardGrazian/BallTracking.git
> cd BallTracking
> python3 -m venv bt
> source bt/bin/activate
(bt) > pip install --upgrade pip 
(bt) > pip install -r requirements.txt
(bt) > python main.py --input <input-dir> --output <output-dir>
```

# Further work
Although the code in this repo produces decent results for a first pass, it is not ready to support production use cases

The main weakness of the current implementation is that it relies on a pre-trained Mask R-CNN to detect objects in each frame of the video. 
This directly leads to 2 types of error:
* False negatives
  * The ball is visible in the video as it darts across the screen, but it is too faint to detect in any individual frame
  * This is by far the most common error in the current version
* False positives
  * A non-ball object is detected for a single frame of the video
 
We can address both these errors by training a detector on video directly. 
Since the Mask R-CNN can already detect the ball in the majority of cases, we can extract features from that model. 
Then our video detector could consume those features rather than the raw frames
