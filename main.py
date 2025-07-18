import os
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import yt_dlp
from yt_dlp import YoutubeDL
from urllib.parse import urlparse, parse_qs
from pydantic import BaseModel
import assemblyai as aai
import uvicorn
import tempfile
import imageio
import argparse
import uuid
import subprocess
import shutil


app = FastAPI()


STATIC_DIR = os.path.join(os.getcwd(), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


YOUTUBE_COOKIES = os.getenv("YOUTUBE_COOKIES", "")

def create_cookie_file():
    """
    Create a temporary cookie file from environment variable
    Expected format: Netscape format cookies as a multi-line string
    """
    if not YOUTUBE_COOKIES:
        return None
    
    try:

        cookie_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        
   
        cookie_file.write(YOUTUBE_COOKIES)
        cookie_file.flush()
        cookie_file.close()
        
        return cookie_file.name
    except Exception as e:
        print(f"Warning: Failed to create cookie file: {e}")
        return None

def get_ydl_opts():
    """
    Get YoutubeDL options with cookies from environment variables
    """
    base_opts = {
        'quiet': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }
    

    cookie_file = create_cookie_file()
    if cookie_file:
        base_opts['cookiefile'] = cookie_file
    
    return base_opts

def cleanup_cookie_file(ydl_opts):
    """
    Clean up temporary cookie file if it was created
    """
    if 'cookiefile' in ydl_opts:
        try:
            os.unlink(ydl_opts['cookiefile'])
        except Exception:
            pass  

def extract_video_info(url: str):
    ydl_opts = {
        **get_ydl_opts(),
        'extract_flat': False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'Unknown Title')
            formats = info.get('formats', [])
            
 
            thumbnails = info.get('thumbnails', [])
            best_thumbnail = None
            max_width = 0
            
            for thumb in thumbnails:
                width = thumb.get('width', 0)
                if width > max_width:
                    max_width = width
                    best_thumbnail = thumb.get('url')
            

            default_thumbnail = info.get('thumbnail', None)

   
            audio_video_formats = [
                {
                    'format_id': fmt['format_id'],
                    'resolution': fmt.get('resolution', 'N/A'),
                    'filesize': fmt.get('filesize') or fmt.get('filesize_approx'),
                    'url': fmt['url']
                }
                for fmt in formats if fmt.get('acodec') != 'none' and fmt.get('vcodec') != 'none'
            ]


            highest_audio = max(
                (fmt for fmt in formats if fmt.get('acodec') != 'none' and fmt.get('vcodec') == 'none' and fmt.get('ext') == 'm4a'),
                key=lambda x: x.get('abr', 0),
                default=None
            )

            highest_audio_info = {
                'format_id': highest_audio['format_id'],
                'bitrate': highest_audio.get('abr', 'Unknown'),
                'filesize': highest_audio.get('filesize') or highest_audio.get('filesize_approx'),
                'url': highest_audio['url']
            } if highest_audio else None

            result = {
                "title": title,
                "formats": audio_video_formats,
                "highest_audio": highest_audio_info,
                "thumbnail": best_thumbnail or default_thumbnail
            }
            
        
            cleanup_cookie_file(ydl_opts)
            
            return result
        except Exception as e:
            cleanup_cookie_file(ydl_opts)
            raise HTTPException(status_code=500, detail=f"Error extracting video info: {str(e)}")

def search_video_by_title(title: str):
    ydl_opts = {
        **get_ydl_opts(),
        'extract_flat': True,
        'force_generic_extractor': True,  
    }

    with YoutubeDL(ydl_opts) as ydl:
        try:
            search_results = ydl.extract_info(f"ytsearch:{title}", download=False)
            if not search_results or 'entries' not in search_results:
                cleanup_cookie_file(ydl_opts)
                raise HTTPException(status_code=404, detail="No videos found for the given title.")

        
            first_entry = search_results['entries'][0]
            result = first_entry['url']
            
        
            cleanup_cookie_file(ydl_opts)
            
            return result
        except Exception as e:
            cleanup_cookie_file(ydl_opts)
            raise HTTPException(status_code=500, detail=f"Error searching video by title: {str(e)}")

def extract_playlist_id(url: str) -> str:
    """
    Extracts the playlist ID from a YouTube URL.
    """
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    playlist_id = query_params.get('list', [None])[0]
    if not playlist_id:
        raise HTTPException(status_code=400, detail="No playlist ID found in the URL.")
    return playlist_id

def extract_playlist_video_urls(playlist_url: str):
    """
    Extracts video URLs and metadata from a YouTube playlist.
    """
  
    if "watch?v=" in playlist_url and "list=" in playlist_url:
        playlist_id = extract_playlist_id(playlist_url)
        playlist_url = f"https://www.youtube.com/playlist?list={playlist_id}"

    ydl_opts = {
        **get_ydl_opts(),
        'extract_flat': True,
        'force_generic_extractor': True,  
        'extractor_args': {
            'youtubetab': {'skip': ['authcheck']}
        }
    }

    with YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(playlist_url, download=False)
            if not info or 'entries' not in info:
                raise HTTPException(status_code=404, detail="No videos found in the playlist.")

            video_urls = [
                {
                    'title': entry.get('title', 'Unknown Title'),
                    'url': entry.get('url') or entry.get('webpage_url', 'N/A'),  
                    'thumbnail': entry.get('thumbnail', None)  
                }
                for entry in info['entries'] if entry
            ]

            return {
                "playlist_title": info.get('title', 'Unknown Playlist'),
                "playlist_thumbnail": info.get('thumbnail', None),  
                "video_urls": video_urls
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error extracting playlist video URLs: {str(e)}")

def download_and_merge_video(url: str, resolution: str):
    """
    Downloads video and audio separately and merges with ffmpeg.
    Returns the path to the merged file and thumbnail URL.
    
    If the specified resolution (720p/1080p) isn't available, falls back to best available quality.
    """
    try:
     
        file_id = str(uuid.uuid4())
        output_dir = os.path.join(STATIC_DIR, file_id)
        os.makedirs(output_dir, exist_ok=True)
        
     
        common_opts = get_ydl_opts()
        
     
        with YoutubeDL({**common_opts, 'noplaylist': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'video')
            
         
            thumbnails = info.get('thumbnails', [])
            best_thumbnail = None
            max_width = 0
            
            for thumb in thumbnails:
                width = thumb.get('width', 0)
                if width > max_width:
                    max_width = width
                    best_thumbnail = thumb.get('url')
            
            if not best_thumbnail:
                best_thumbnail = info.get('thumbnail', None)
            
            target_height = int(resolution.replace("p", ""))
            actual_resolution = resolution 
        
        title = "".join(c for c in title if c.isalnum() or c in [' ', '-', '_']).strip()
        title = title.replace(' ', '_')
        
        output_path = os.path.join(STATIC_DIR, f"{title}_{resolution}_{file_id}.mp4")
        
        video_path = os.path.join(output_dir, f"{file_id}_video.mp4")
        audio_path = os.path.join(output_dir, f"{file_id}_audio.m4a")
        
    
        if resolution == "720p":
          
            video_format = (
                "bestvideo[height<=720][ext=mp4]/bestvideo[height<=720][ext=webm]/"
                "bestvideo[height<=720]/best[height<=720][ext=mp4]/best[height<=720][ext=webm]/"
                "best[height<=720]/bestvideo[height<=800]/best[height<=800]/"
                "bestvideo/best"
            )
            height_filter = 720
        else:  

            video_format = (
                "bestvideo[height<=1080][ext=mp4]/bestvideo[height<=1080][ext=webm]/"
                "bestvideo[height<=1080]/best[height<=1080][ext=mp4]/best[height<=1080][ext=webm]/"
                "best[height<=1080]/bestvideo[height<=1200]/best[height<=1200]/"
                "bestvideo/best"
            )
            height_filter = 1080
        
    
        video_opts = {
            **common_opts,
            'format': video_format,
            'outtmpl': video_path,
            'format_sort': ['res', 'ext:mp4:m4a', 'ext:webm'],  
        }
        
       
        video_downloaded = False
        fallback_formats = [
            video_format,  
            f"best[height<={height_filter}]",  
            "bestvideo/best", 
            "best"  
        ]
        
        for fmt in fallback_formats:
            try:
                video_opts['format'] = fmt
                with YoutubeDL(video_opts) as ydl:
                    ydl.download([url])
                video_downloaded = True
                break
            except Exception as e:
                print(f"Video download failed with format '{fmt}': {e}")
                continue
        
        if not video_downloaded:
            raise Exception("Failed to download video with any format")
            
      
        audio_downloaded = False
        temp_audio_path = os.path.join(output_dir, f"{file_id}_temp_audio")
        
     
        audio_strategies = [
            
            {
                'format': 'bestaudio/best',
                'outtmpl': temp_audio_path,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'm4a',
                    'preferredquality': '192',
                }],
            },
          
            {
                'format': 'bestaudio[ext=m4a]/bestaudio[ext=aac]/bestaudio[ext=mp3]/bestaudio',
                'outtmpl': audio_path,
            },
        
            {
                'format': 'bestaudio',
                'outtmpl': audio_path,
            },
         
            {
                'format': 'best',
                'outtmpl': temp_audio_path,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'm4a',
                    'preferredquality': '192',
                }],
            }
        ]
        
        for i, strategy in enumerate(audio_strategies):
            try:
                audio_opts = {**common_opts, **strategy}
                with YoutubeDL(audio_opts) as ydl:
                    ydl.download([url])
                

                if 'postprocessors' in strategy:
                    audio_files = [f for f in os.listdir(output_dir) if 'temp_audio' in f]
                    if audio_files:
                        actual_audio_path = os.path.join(output_dir, audio_files[0])
                        shutil.move(actual_audio_path, audio_path)
                        audio_downloaded = True
                        break
                else:
                 
                    if os.path.exists(audio_path):
                        audio_downloaded = True
                        break
                        
            except Exception as e:
                print(f"Audio download strategy {i+1} failed: {e}")
                continue
        
        if not audio_downloaded:
            raise Exception("Failed to download audio with any strategy")
        
    
        probe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=p=0',
            video_path
        ]
        
        try:
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            width, height = map(int, probe_result.stdout.strip().split(','))
            actual_resolution = f"{height}p"
        except Exception:
          
            pass
        
     
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0',
            output_path
        ]
        

        subprocess.run(cmd, check=True)
        
      
        if actual_resolution != resolution:
            new_output_path = os.path.join(STATIC_DIR, f"{title}_{actual_resolution}_{file_id}.mp4")
            os.rename(output_path, new_output_path)
            output_path = new_output_path
        
   
        shutil.rmtree(output_dir)
        

        relative_path = os.path.basename(output_path)
        return relative_path, best_thumbnail
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

import os
import uuid
import shutil
import requests
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4
from mutagen.id3 import ID3, APIC, TIT2, TPE1, TALB
from yt_dlp import YoutubeDL

def download_audio(url: str, audio_format: str):
    """
    Downloads audio from a video URL in the specified format and adds metadata.
    Returns the path to the downloaded audio file.
    
    Parameters:
    - url: URL of the video
    - audio_format: Format of the audio (mp3 or m4a)
    """
    try:
        file_id = str(uuid.uuid4())
        output_dir = os.path.join(STATIC_DIR, file_id)
        os.makedirs(output_dir, exist_ok=True)
        
        common_opts = get_ydl_opts()
        
   
        with YoutubeDL({**common_opts, 'noplaylist': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'audio')
            uploader = info.get('uploader', 'Unknown Artist')
            channel = info.get('channel', uploader)
            description = info.get('description', '')
            
           
            thumbnails = info.get('thumbnails', [])
            best_thumbnail = None
            max_width = 0
            
            for thumb in thumbnails:
                width = thumb.get('width', 0)
                if width > max_width:
                    max_width = width
                    best_thumbnail = thumb.get('url')
            
            if not best_thumbnail:
                best_thumbnail = info.get('thumbnail', None)
        
       
        clean_title = "".join(c for c in title if c.isalnum() or c in [' ', '-', '_']).strip()
        clean_title = clean_title.replace(' ', '_')
        
      
        temp_audio_path = os.path.join(output_dir, f"{file_id}_temp.{audio_format}")
        output_path = os.path.join(STATIC_DIR, f"{clean_title}_{audio_format}_{file_id}.{audio_format}")
        thumbnail_path = os.path.join(output_dir, f"{file_id}_thumb.jpg")
        
       
        thumbnail_data = None
        if best_thumbnail:
            try:
                response = requests.get(best_thumbnail, timeout=10)
                if response.status_code == 200:
                    with open(thumbnail_path, 'wb') as f:
                        f.write(response.content)
                    thumbnail_data = response.content
            except Exception as e:
                print(f"Failed to download thumbnail: {e}")
        
      
        if audio_format == "mp3":
            format_option = "bestaudio/best"
            postprocessors = [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }]
        else:  
            format_option = "bestaudio[ext=m4a]/bestaudio"
            postprocessors = [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'm4a',
                'preferredquality': '192',
            }]
        
        audio_opts = {
            **common_opts,
            'format': format_option,
            'outtmpl': temp_audio_path,
            'postprocessors': postprocessors,
        }
        
        with YoutubeDL(audio_opts) as ydl:
            ydl.download([url])
        
      
        downloaded_files = os.listdir(output_dir)
        audio_files = [f for f in downloaded_files if f.endswith(('.mp3', '.m4a'))]
        
        if not audio_files:
            raise Exception("No audio file was downloaded")
        
        downloaded_file = os.path.join(output_dir, audio_files[0])
        shutil.move(downloaded_file, output_path)
        
     
        add_metadata(output_path, audio_format, title, uploader, channel, thumbnail_data)
        
       
        shutil.rmtree(output_dir)
        
        relative_path = os.path.basename(output_path)
        return relative_path, best_thumbnail
        
    except Exception as e:
    
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


def add_metadata(file_path: str, audio_format: str, title: str, artist: str, album: str, thumbnail_data: bytes = None):
    """
    Adds metadata to the audio file including thumbnail as cover art.
    
    Parameters:
    - file_path: Path to the audio file
    - audio_format: Format of the audio (mp3 or m4a)
    - title: Track title
    - artist: Artist name
    - album: Album name (using channel name)
    - thumbnail_data: Binary data of the thumbnail image
    """
    try:
        if audio_format == "mp3":
          
            audio = MP3(file_path, ID3=ID3)
            
           
            if audio.tags is None:
                audio.add_tags()
            
         
            audio.tags.add(TIT2(encoding=3, text=title)) 
            audio.tags.add(TPE1(encoding=3, text=artist)) 
            audio.tags.add(TALB(encoding=3, text=album))  
            
         
            if thumbnail_data:
                audio.tags.add(APIC(
                    encoding=3,  
                    mime='image/jpeg',
                    type=3,
                    desc='Cover',
                    data=thumbnail_data
                ))
            
            audio.save()
            
        elif audio_format == "m4a":
          
            audio = MP4(file_path)
            
        
            audio['\xa9nam'] = [title]  
            audio['\xa9ART'] = [artist] 
            audio['\xa9alb'] = [album]   
            

            if thumbnail_data:
                audio['covr'] = [thumbnail_data]
            
            audio.save()
            
    except Exception as e:
        print(f"Failed to add metadata: {e}")


@app.get("/download-audio")
async def download_audio_endpoint(
    query: str = Query(..., description="URL or title of the video"),
    format: str = Query("mp3", description="Audio format (mp3 or m4a)")
):
    try:
       
        if format not in ["mp3", "m4a"]:
            raise HTTPException(status_code=400, detail="Format must be either mp3 or m4a")
        
     
        parsed_url = urlparse(query)
        if parsed_url.scheme and parsed_url.netloc:
            video_url = query
        else:
            video_url = search_video_by_title(query)
        
       
        output_file, thumbnail_url = download_audio(video_url, format)
        
       
        file_url = f"/static/{output_file}"
        server_url = os.getenv("SERVER_URL", "https://kanade710-ytt.hf.space")
        download_url = f"{server_url}{file_url}"
        
        return JSONResponse(content={
            "download_url": download_url,
            "filename": output_file,
            "thumbnail": thumbnail_url
        })
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/video-info")
async def get_video_info(query: str = Query(..., description="URL or title of the video")):
    try:
      
        parsed_url = urlparse(query)
        if parsed_url.scheme and parsed_url.netloc:
            video_url = query
        else:
            video_url = search_video_by_title(query)

        video_data = extract_video_info(video_url)
        return JSONResponse(content=video_data)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download-video")
async def download_video(
    query: str = Query(..., description="URL or title of the video"),
    resolution: str = Query("720p", description="Video resolution (720p or 1080p)")
):
    try:
     
        if resolution not in ["720p", "1080p"]:
            raise HTTPException(status_code=400, detail="Resolution must be either 720p or 1080p")
        

        parsed_url = urlparse(query)
        if parsed_url.scheme and parsed_url.netloc:
            video_url = query
        else:
            video_url = search_video_by_title(query)
        
      
        output_file, thumbnail_url = download_and_merge_video(video_url, resolution)
        

        actual_resolution = resolution
        if "_best_" in output_file:
            actual_resolution = "best quality available"
        elif "_" in output_file:
            parts = output_file.split("_")
            for part in parts:
                if part.endswith("p") and part[:-1].isdigit():
                    actual_resolution = part
                    break
        
     
        file_url = f"/static/{output_file}"
        server_url = os.getenv("SERVER_URL", "https://kanade710-ytt.hf.space")
        download_url = f"{server_url}{file_url}"
        
        return JSONResponse(content={
            "download_url": download_url,
            "filename": output_file,
            "thumbnail": thumbnail_url,
            "actual_resolution": actual_resolution
        })
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/direct-download/{filename}")
async def get_file(filename: str):
    file_path = os.path.join(STATIC_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="video/mp4"
    )


class TranscriptionRequest(BaseModel):
    audio_url: str

@app.get("/transcribe")
async def transcribe_audio(audio_url: str = Query(..., description="Publicly accessible URL for audio file")):
    try:
        config = aai.TranscriptionConfig(
            speaker_labels=True,
            speech_model=aai.SpeechModel.nano,
            language_detection=True
        )
        transcript = transcriber.transcribe(audio_url, config)

        if transcript.status == aai.TranscriptStatus.error:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {transcript.error}")

        return {"transcription": transcript.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/playlist-download-urls")
async def get_playlist_video_urls(playlist_url: str = Query(..., description="URL of the YouTube playlist")):
    try:
        result = extract_playlist_video_urls(playlist_url)
        return JSONResponse(content=result)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def process_and_upload_webp(url: str) -> str:
    try:

        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Invalid URL")


        response = requests.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download the file")

 
        with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as webp_file:
            webp_file.write(response.content)
            webp_file_path = webp_file.name


        output_path = webp_file_path.replace(".webp", ".mp4")
        reader = imageio.get_reader(webp_file_path)
        writer = imageio.get_writer(output_path, fps=30)
        for frame in reader:
            writer.append_data(frame)
        writer.close()

  
        with open(output_path, 'rb') as mp4_file:
            upload_response = requests.post(
                'https://uguu.se/upload',
                files={"files[]": mp4_file}
            )
            if upload_response.status_code != 200:
                raise HTTPException(status_code=500, detail="Failed to upload the file")

            file_url = upload_response.json()["files"][0]["url"]


        os.remove(webp_file_path)
        os.remove(output_path)

        return file_url
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/convert")
async def convert_webp_to_mp4(url: str):
    file_url = await process_and_upload_webp(url)
    return JSONResponse(content={"url": file_url})


def cli():
    parser = argparse.ArgumentParser(description="Convert and upload WEBP to MP4.")
    parser.add_argument("--url", type=str, required=True, help="URL of the .webp file.")
    args = parser.parse_args()


    import asyncio
    result = asyncio.run(process_and_upload_webp(args.url))
    print(f"Uploaded file link: {result}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)