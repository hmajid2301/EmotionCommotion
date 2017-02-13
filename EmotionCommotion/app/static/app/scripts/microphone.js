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







