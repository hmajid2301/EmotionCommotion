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
    interval = setInterval(loop, 1000)
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
    if (frameNum > 0) {
        var data = new FormData();
        data.append("enctype", "multipart/form-data");
        data.append("blob", blob);
        data.append("frame-number", frameNum);
        console.log(frameNum);

        $.ajax({
            url: "/blob",
            type: "post",
            data: data,
            processData: false,
            contentType: false,
            success: function (a) {
                $("#graph").show()
                console.log(a);
                emotion = a;
            },
            error: function (e) {
                console.log(e)
            }
        });

    }
    frameNum++;
}


function loop() {
    audioRecorder.getBuffers(gotBuffers);
}
