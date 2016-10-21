var wavesurfer = WaveSurfer.create({ container: '#waveform', waveColor: 'violet' });

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

    if ($('#microphone').is(':visible')) {
        $("#microphone").hide()
        $("#waveform").show()
        $("#pause").show()
        $("#stop").show()
    }
});
