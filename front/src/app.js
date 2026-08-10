import { videoPaths } from './videoPaths';

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('search-button').addEventListener('click', function() {
        const query = document.getElementById('search-bar').value.toLowerCase();
        const results = videoPaths.filter(path => path.toLowerCase().includes(query));
        
        const resultsContainer = document.getElementById('results');
        resultsContainer.innerHTML = '';

        results.forEach((path, _) => {
            const resultItem = document.createElement('div');
            const game_path = path.split('SoccerNet_in_lrlab/')[1];
            resultItem.textContent = parseMatchInfo(game_path);
            resultItem.className = 'result-item';
            resultItem.addEventListener('click', () => {
                selectVideo(path);
                const extractedInfo = getVideoInfo(game_path);
                console.log(extractedInfo);
                document.getElementById('game').value = extractedInfo.game;
                document.getElementById('half').value = extractedInfo.number;
            });
            resultsContainer.appendChild(resultItem);
        });

        if (results.length === 0) {
            resultsContainer.innerHTML = '<div>No results found</div>';
        }
    });

    document.getElementById('jump-button').addEventListener('click', function() {
        const timeInput = document.getElementById('time-input').value;
        const videoElement = document.getElementById('my-video');

        if (videoElement && timeInput) {
            const timeInSeconds = parseTimeInput(timeInput);
            if (timeInSeconds !== null) {
                videoElement.currentTime = timeInSeconds;
                videoElement.play();
            } else {
                alert('Invalid time format');
            }
        }
    });

    document.getElementById('rewind-button').addEventListener('click', function() {
        const videoElement = document.getElementById('my-video');
        if (videoElement) {
            videoElement.currentTime = Math.max(0, videoElement.currentTime - 10);
        }
    });

    document.getElementById('forward-button').addEventListener('click', function() {
        const videoElement = document.getElementById('my-video');
        if (videoElement) {
            videoElement.currentTime = Math.min(videoElement.duration, videoElement.currentTime + 10);
        }
    });
});
function selectVideo(videoPath) {
    // (event trigger) Load the selected video
    const videoElement = document.getElementById('my-video');
    const sourceElement = document.getElementById('video-source');
    const trackElement = videoElement.querySelector('track');

    let vttPath = "";
    if (videoPath.includes("1_224p.mkv")) {
        vttPath = videoPath.replace("1_224p.mkv", "1_asr.vtt");
    } else if (videoPath.includes("1_720p.mkv")) {
        vttPath = videoPath.replace("1_720p.mkv", "1_asr.vtt");
    } else if (videoPath.includes("2_224p.mkv")) {
        vttPath = videoPath.replace("2_224p.mkv", "2_asr.vtt");
    } else if (videoPath.includes("2_720p.mkv")) {
        vttPath = videoPath.replace("2_720p.mkv", "2_asr.vtt");
    }
    
    trackElement.src = vttPath;

    sourceElement.src = videoPath;
    videoElement.load();

    // Display the current match path
    const matchInfo = parseMatchInfo(videoPath.split('SoccerNet_in_lrlab/')[1]);
    document.getElementById('current-match').textContent = matchInfo;
}

function parseTimeInput(input) {
    // Parse time in the format MM:SS or MM:SS:MS
    // return time in seconds or null if invalid
    const parts = input.split(':');
    if (parts.length === 2 || parts.length === 3) {
        const minutes = parseInt(parts[0], 10);
        const seconds = parseInt(parts[1], 10);
        if (!isNaN(minutes) && !isNaN(seconds)) {
            return minutes * 60 + seconds;
        }
    }
    return null;
}

document.getElementById('videoForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    // ローディング表示を開始
    const loadingIndicator = document.getElementById('loading-indicator');
    loadingIndicator.style.display = 'block';


    const formData = new FormData(event.target);
    
    // textのstart(mm:ss)を numberのstart(seconds)に変換
    const start = formData.get('start');
    const end = formData.get('end');
    formData.set('start', parseTimeInput(start));
    formData.set('end', parseTimeInput(end));

    console.log("start:", start);
    console.log("end:", end);
    console.log("game:", formData.get('game'));
    console.log("half:", formData.get('half'));

    const response = await fetch('http://127.0.0.1:8000/generate/', {
        method: 'POST',
        body: formData
    });

    try {
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const downloadLink = document.getElementById('videoDownload');
            downloadLink.href = url;
            downloadLink.style.display = 'block';
            document.getElementById('downloadLink').style.display = 'block';
        } else {
            // server側のエラーメッセージを表示
            const errorText = await response.text();
            const overallErrorText = `Failed to generate video: ${response.status} ${response.statusText}`;
            alert(overallErrorText);
        } 
    } finally {
        // ローディング表示を終了
        loadingIndicator.style.display = 'none';
    }
});

function parseMatchInfo(relativePath) {
  const parts = relativePath.split("/");

  // parts[0] がリーグ名, parts[1] がシーズン, parts[2] が試合情報, parts[3] がハイライトファイル名
  const league = parts[0];
  const season = parts[1];
  const matchInfo = parts[2];
  const highlightFile = parts[3];

  // 試合情報を正規表現でパース
  //   例: "2016-08-20 - 17-00 Burnley 2 - 0 Liverpool"
  const matchPattern = /^(\d{4}-\d{2}-\d{2}) - (\d{2}-\d{2}) (.+?) (\d+) - (\d+) (.+)$/;
  const matchResult = matchPattern.exec(matchInfo);

  if (!matchResult) {
    throw new Error("試合情報の形式が想定と異なります: " + matchInfo);
  }

  // 日付・時間・チームなどを抽出
  const date      = matchResult[1];
  // const time      = matchResult[2];
  const homeTeam  = matchResult[3];
  // const homeScore = matchResult[4];
  // const awayScore = matchResult[5];
  const awayTeam  = matchResult[6];

  // ハーフ情報(前半/後半)の抽出
  //   例: "1_224p.mkv" => "1" -> 1st half
  const baseName = highlightFile.replace(/\.[^.]+$/, "");
  const half = baseName.split("_")[0];

  let halfText;
  if (half === "1") {
    halfText = "1st half";
  } else if (half === "2") {
    halfText = "2nd half";
  } else {
    // 必要に応じて追加の処理(延長戦等)を入れる
    halfText = half;
  }

  // 出力形式に合わせて文字列を生成して返す
  // "league season date homeTeam - awayTeam (1st half or 2nd half)"
  return `${league} ${season} ${date} ${homeTeam} - ${awayTeam} (${halfText})`;
}


function getVideoInfo(relativePath) {
  // 最後のスラッシュの位置を取得
  const lastSlashIndex = relativePath.lastIndexOf('/');
  // ディレクトリ部分を取得
  const game = relativePath.substring(0, lastSlashIndex);
  // ファイル名部分を取得
  const filename = relativePath.substring(lastSlashIndex + 1);
  // アンダースコアで分割して先頭の部分（数字）を抽出
  const number = filename.split('_')[0];
  
  return { game, number };
}
