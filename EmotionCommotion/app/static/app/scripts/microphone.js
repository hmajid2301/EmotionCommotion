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
<<<<<<< HEAD
    interval = setInterval(loop, 1000)
=======
    interval = setInterval(loop, 500)
>>>>>>> 8f4780250f7a4bc0c2be7b3633f3acbb29150cb8

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

lastblob=null;
function doneEncoding(blob) {
    if (frameNum > 0) {
        var data = new FormData();
    
        data.append("enctype", "multipart/form-data");
        data.append("data", [lastblob, blob]);
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
    frameNum++;
<<<<<<< HEAD

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
            console.log(a)
        },
        error: function (e) {
            console.log(e)
        }
    });
=======
    lastblob = blob;
>>>>>>> 8f4780250f7a4bc0c2be7b3633f3acbb29150cb8
}


function loop() {
    console.log("Stopping")
    toggleRecording(document.getElementById("microphone"))
    console.log("Starting")
    toggleRecording(document.getElementById("microphone"))
}
