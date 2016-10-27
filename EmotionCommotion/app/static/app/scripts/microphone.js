
'use strict'

var wavesurfer = WaveSurfer.create({
    container: '#waveform',
    waveColor: '#367ee9'
});

var microphone = Object.create(WaveSurfer.Microphone);

microphone.init({
    wavesurfer: wavesurfer
});

microphone.on('deviceReady', function (stream) {
    console.log('Device ready!', stream);
});
microphone.on('deviceError', function (code) {
    console.warn('Device error: ' + code);
});

let log = console.log.bind(console),
  id = val => document.getElementById(val),
  ul = id('ul'),
  gUMbtn = id('start'),
  start = id('start'),
  stop = id('stop'),
  stream,
  recorder,
  counter=1,
  chunks,
  media;


window.onload = function() {

  let mv = id('mediaVideo'),
  mediaOptions = {
          tag: 'audio',
          type: 'audio/ogg',
          ext: '.ogg',
          gUM: { audio: true }
  };
  media = mediaOptions;
  navigator.mediaDevices.getUserMedia(media.gUM).then(_stream => {
    stream = _stream;
    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = e => {
      chunks.push(e.data);
      if(recorder.state == 'inactive')  makeLink();
    };
    log('got media successfully');
  }).catch(log);
}


$("#microphone").click(function () {
    $("#microphone").hide()
    $("#waveform").show()
    $("#graph").show()
    $(".toggle.btn").show()
    $("#stop").show()
    recorder.start();
    chunks = [];
    microphone.start()
});


$("#stop").click(function () {
    $("#microphone").show()
    $("#waveform").hide()
    $("#stop").hide()
    $("#graph").hide()
    $(".toggle.btn").hide()
    $("#emojis").hide()
    microphone.stop()
    recorder.stop();
});


function makeLink(){
  let blob = new Blob(chunks, {type: media.type })
    , url = URL.createObjectURL(blob)
    , li = document.createElement('li')
    , mt = document.createElement(media.tag)
    , hf = document.createElement('a')
  ;
  mt.controls = true;
  mt.src = url;
  hf.href = url;
  li.appendChild(mt);
  li.appendChild(hf);
  ul.appendChild(li);
}

