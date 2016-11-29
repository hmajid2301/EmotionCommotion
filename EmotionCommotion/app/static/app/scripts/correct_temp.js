$(".emoji").click(function(){
    $("#choice").show("100");
});


$(".fa-check").click(function () {
    $("#choice").hide("100");
});

$(".fa-times").click(function () {
    $("#choice .fa").hide("100");
    $(".correct").show("100");
});

var emotion;
$(".correct #1").click(function () {
    emotion = "anger";
    $("#choice").hide("100");
});
$(".correct #2").click(function () {
    emotion = "sad";
    $("#choice").hide("100");
});
$(".correct #3").click(function () {
    emotion = "neutral";
    $("#choice").hide("100");
});
$(".correct #4").click(function () {
    emotion = "happy";
    $("#choice").hide("100");
});