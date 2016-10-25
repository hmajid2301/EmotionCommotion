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

$("#microphone").click(function () {
    $("#microphone").hide()
    $("#waveform").show()
    $("#stop").show()
});


$("#stop").click(function () {
    $("#microphone").show()
    $("#waveform").hide()
    $("#stop").hide()
    microphone.stop()
});
