## Testing locally

If you want to open the HTML pages in this folder, you will need the
bebras-modules repository cloned ; this can be done with the command

```
git submodules update --init
```

## Adding a new video

To add a new video, create a copy of another folder and edit the index.html
file with :
* an unique `id`
* the `video_id` of the video on YouTube (from the original URL)
* optionally, the sections of the video, to be displayed on the right
