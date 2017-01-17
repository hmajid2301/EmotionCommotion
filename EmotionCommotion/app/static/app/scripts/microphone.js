var emotion;

var wavesurfer = WaveSurfer.create({
    container: '#waveform',
    waveColor: '#367ee9'
});

var microphone = Object.create(WaveSurfer.Microphone);

microphone.init({
    wavesurfer: wavesurfer
});


function GetEmotion() {

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

function doneEncoding(blob) {
    var data = new FormData();

    data.append("fname", "test.wav");
    data.append("recording", "recording");
    data.append("enctype", "multipart/form-data");
    data.append("data", blob);

    $.ajax({
        url: "/blob",
        type: "post",
        data: data,
        processData: false,
        contentType: false,
        success: function (a) {
            emotion = a.emotion
            $("#emojis").show()
        },
        error: function (e) {
            console.log(e)
        }
    });
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


function saveAudio() {
    audioRecorder.exportWAV(doneEncoding);
}

function gotBuffers(buffers) {
    var canvas = document.getElementById("wavedisplay");

    audioRecorder.exportWAV(doneEncoding);
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


function gotStream(stream) {
    inputPoint = audioContext.createGain();

    // Create an AudioNode from the stream.
    realAudioInput = audioContext.createMediaStreamSource(stream);
    audioInput = realAudioInput;
    audioInput.connect(inputPoint);


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







