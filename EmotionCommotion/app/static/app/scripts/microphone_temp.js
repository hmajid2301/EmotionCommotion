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
       type: 'audio/wav',
       ext: '.wav',
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
  }).catch(log);
}

function makeLink() {

    //shdkasdkdhas

    var blob = new Blob(chunks, { type: media.type });
    var url = URL.createObjectURL(blob);
    var data = new FormData();

    data.append("fname", "test.wav");
    data.append("enctype", "multipart/form-data");
    data.append("data", blob);

    $.ajax({
        url: "/blob",
        type: "POST",
        data: data,
        processData: false,
        contentType: false,
        success: function (a) {
            console.log("Ajax", a.emotion)
            emotion = a.emotion
            $("#emojis").show()
        },
        error: function () {
            console.log("ERROR")
        }
    });
}

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


$("#microphone").click(function () {
    $("#microphone").hide()
    $("#emojis").hide()
    $("#waveform").show()
    $("#stop").show()
    recorder.start();
    microphone.start()
    chunks = [];
});


$("#stop").click(function () {
    $("#microphone").show()
    $("#waveform").hide()
    $("#stop").hide()
    microphone.stop()
    recorder.stop();
});


