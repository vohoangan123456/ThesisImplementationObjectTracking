def merge_two_video(video_path1, video_path2):
    import imageio
    imageio.plugins.ffmpeg.download()
    from moviepy.editor import VideoFileClip, clips_array
    from random import randint
    import os
    rd_number = randint(0, 1000) # random number for save video
    clip1 = VideoFileClip(video_path1)
    clip2 = VideoFileClip(video_path2)
    final_clip = clips_array([[clip1, clip2]])
    final_clip.resize(width=1200).write_videofile("./videos/{0}_{1}_{2}.mp4".format(os.path.basename(video_path1), os.path.basename(video_path2), str(rd_number)))
