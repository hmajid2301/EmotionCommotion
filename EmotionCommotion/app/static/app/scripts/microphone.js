var emotion;
var interval = null;
var frameNum = 0;

var wavesurfer = WaveSurfer.create({
    container: '#waveform',
    waveColor: '#367ee9'
});

var microphone = Object.create(WaveSurfer.Microphone);

microphone.init({
    wavesurfer: wavesurfer
});

function GetEmotion() {
    return emotion;
}


$("#microphone").click(function () {
    console.log("Microphone Clicked")

    $("#microphone").hide()
    $("#emojis").hide()
    $("#waveform").show()
    $("#stop").show()
    toggleRecording(this)
    microphone.start()
    interval = setInterval(loop, 2000)

});


$("#stop").click(function () {
    console.log("Stop Clicked")

    clearInterval(interval);
    $("#microphone").show()
    $("#waveform").hide()
    $("#stop").hide()
    toggleRecording(document.getElementById("microphone"))
    microphone.stop()
});

function doneEncoding(blob) {
    var data = new FormData();
    frameNum++;

    data.append("fname", "test.wav");
    data.append("enctype", "multipart/form-data");
    data.append("data", blob);
    data.append("frame-number", frameNum);
    console.log(frameNum);


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


function loop() {
    console.log("Stopping")
    toggleRecording(document.getElementById("microphone"))
    setTimeout(function () { console.log("waiting");}, 200)
}

