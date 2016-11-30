var emotion;

var wavesurfer = WaveSurfer.create({
    container: '#waveform',
    waveColor: '#367ee9'
});

var microphone = Object.create(WaveSurfer.Microphone);

microphone.init({
    wavesurfer: wavesurfer
});

//microphone.on('deviceReady', function (stream) {
//    console.log('Device ready!', stream);
//});
//microphone.on('deviceError', function (code) {
//    console.warn('Device error: ' + code);
//});

//let log = console.log.bind(console),
//  id = val => document.getElementById(val),
//  ul = id('ul'),
//  gUMbtn = id('start'),
//  start = id('start'),
//  stop = id('stop'),
//  stream,
//  recorder,
//  counter=1,
//  chunks,
//  media;


//window.onload = function() {

//   let mv = id('mediaVideo'),
//   mediaOptions = {
//       tag: 'audio',
//       type: 'audio/wav',
//       ext: '.wav',
//       gUM: { audio: true }
//   };

//  media = mediaOptions;
//  navigator.mediaDevices.getUserMedia(media.gUM).then(_stream => {
//    stream = _stream;
//    recorder = new MediaRecorder(stream);
//    recorder.ondataavailable = e => {
//      chunks.push(e.data);
//      if(recorder.state == 'inactive')  makeLink();
//    };
//  }).catch(log);
//}

//function makeLink() {

//    //shdkasdkdhas

//    var blob = new Blob(chunks, { type: media.type });
//    var url = URL.createObjectURL(blob);
//    var data = new FormData();

//    data.append("fname", "test.wav");
//    data.append("enctype", "multipart/form-data");
//    data.append("data", blob);

//    $.ajax({
//        url: "/blob",
//        type: "POST",
//        data: data,
//        processData: false,
//        contentType: false,
//        success: function (a) {
//            console.log("Ajax", a.emotion)
//            emotion = a.emotion
//            $("#emojis").show()
//        },
//        error: function () {
//            console.log("ERROR")
//        }
//    });
//}

function GetEmotion() {

    console.log("GetEmotion", emotion)
    switch (emotion) {

        case 'ang':
            return 0;
            break;

        case 'sad':
            return 1;
            break;

        case 'neu':
            return 2;
            break;

        case 'hap':
            return 3;
            break;

        default:
            return -1;
            break
    }
}



/* Copyright 2013 Chris Wilson

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

window.AudioContext = window.AudioContext || window.webkitAudioContext;

var audioContext = new AudioContext();
var audioInput = null,
    realAudioInput = null,
    inputPoint = null,
    audioRecorder = null;
var rafID = null;
var analyserContext = null;
var canvasWidth, canvasHeight;
var recIndex = 0;

/* TODO:

- offer mono option
- "Monitor input" switch
*/

function saveAudio() {
    audioRecorder.exportWAV(doneEncoding);
    // could get mono instead by saying
    // audioRecorder.exportMonoWAV( doneEncoding );
}

function gotBuffers(buffers) {
    var canvas = document.getElementById("wavedisplay");

    //drawBuffer(canvas.width, canvas.height, canvas.getContext('2d'), buffers[0]);

    // the ONLY time gotBuffers is called is right after a new recording is completed - 
    // so here's where we should set up the download.
    audioRecorder.exportWAV(doneEncoding);
}

function doneEncoding(blob) {
    var data = new FormData();

    data.append("fname", "test.wav");
    data.append("enctype", "multipart/form-data");
    data.append("data", blob);

    $.ajax({
        url: "/blob",
        type: "post",
        data: data, 
        processData: false, 
        contentType: false, 
        success: function (a) {
            console.log("ajax", a.emotion)
            emotion = a.emotion
            $("#emojis").show()
        },
        error: function () {
            console.log("error")
        }
    });
}

function toggleRecording(e) {
    if (e.classList.contains("recording")) {
        // stop recording
        audioRecorder.stop();
        e.classList.remove("recording");
        audioRecorder.getBuffers(gotBuffers);
    } else {
        // start recording
        if (!audioRecorder)
            return;
        e.classList.add("recording");
        audioRecorder.clear();
        audioRecorder.record();
    }
}

function convertToMono(input) {
    var splitter = audioContext.createChannelSplitter(2);
    var merger = audioContext.createChannelMerger(2);

    input.connect(splitter);
    splitter.connect(merger, 0, 0);
    splitter.connect(merger, 0, 1);
    return merger;
}

function toggleMono() {
    if (audioInput != realAudioInput) {
        audioInput.disconnect();
        realAudioInput.disconnect();
        audioInput = realAudioInput;
    } else {
        realAudioInput.disconnect();
        audioInput = convertToMono(realAudioInput);
    }

    audioInput.connect(inputPoint);
}

function gotStream(stream) {
    inputPoint = audioContext.createGain();

    // Create an AudioNode from the stream.
    realAudioInput = audioContext.createMediaStreamSource(stream);
    audioInput = realAudioInput;
    audioInput.connect(inputPoint);

    //    audioInput = convertToMono( input );

    analyserNode = audioContext.createAnalyser();
    analyserNode.fftSize = 2048;
    inputPoint.connect(analyserNode);

    audioRecorder = new Recorder(inputPoint);

    zeroGain = audioContext.createGain();
    zeroGain.gain.value = 0.0;
    inputPoint.connect(zeroGain);
    zeroGain.connect(audioContext.destination);
}

function initAudio() {
    if (!navigator.getUserMedia)
        navigator.getUserMedia = navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
    if (!navigator.cancelAnimationFrame)
        navigator.cancelAnimationFrame = navigator.webkitCancelAnimationFrame || navigator.mozCancelAnimationFrame;
    if (!navigator.requestAnimationFrame)
        navigator.requestAnimationFrame = navigator.webkitRequestAnimationFrame || navigator.mozRequestAnimationFrame;

    navigator.getUserMedia(
        {
            "audio": {
                "mandatory": {
                    "googEchoCancellation": "false",
                    "googAutoGainControl": "false",
                    "googNoiseSuppression": "false",
                    "googHighpassFilter": "false"
                },
                "optional": []
            },
        }, gotStream, function (e) {
            alert('Error getting audio');
            console.log(e);
        });
}

window.addEventListener('load', initAudio);


$("#microphone").click(function () {
    $("#microphone").hide()
    $("#emojis").hide()
    $("#waveform").show()
    $("#stop").show()
    toggleRecording(this)
    microphone.start()
});


$("#stop").click(function () {
    $("#microphone").show()
    $("#waveform").hide()
    $("#stop").hide()
    toggleRecording(document.getElementById("microphone"))
    microphone.stop()
});


