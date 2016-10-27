var cv_w = 400, cv_h = 400, cv_r = 150
cv_color = d3.scale.category10();

var cv_arc = d3.svg.arc().outerRadius(cv_r);
var cv_pie = d3.layout.pie().value(function (d) { return d.value });
var cv_svg = d3.select("#graph")
    .append("svg")
    .attr("width", cv_w)
    .attr("height", cv_h)
    .attr("style", "display:block; margin: 0 auto;")
    .append("g")
    .attr("transform", "translate(" + cv_r + "," + cv_r + ")");

colourList = ["rgb(216, 30, 58)", "rgb(123, 204, 24)", "rgb(24, 150, 204)", "rgb(237, 173, 11)"]
emotionList = ["Angry", "Happy", "Sad", "Neutral"]
AddLegend()

function cv_arcTween(a) {
    var i = d3.interpolate(this._current, a);
    this._current = i(0);
    return function (t) {
        return cv_arc(i(t));
    };
}

function AddLegend() {
    var ordinal = d3.scale.ordinal()
      .domain(emotionList)
      .range(colourList);

    var svg = d3.select("svg");

    svg.append("g")
      .attr("class", "legendOrdinal")
      .attr("transform", "translate(310,20)");

    var legendOrdinal = d3.legend.color()
      .shape("path", d3.svg.symbol().type("square").size(200)())
      .shapePadding(10)
      .scale(ordinal);

    svg.select(".legendOrdinal")
      .call(legendOrdinal);
}

function UpdateGraph(data) {
    data = data ? data : {
        "Anger": Math.floor((Math.random() * 10) + 1), "Neutral": Math.floor((Math.random() * 10) + 1)
        , "Happy": Math.floor((Math.random() * 10) + 1)
        , "Sadness": Math.floor((Math.random() * 10) + 1)
    };
    var dataa = d3.entries(data);
    var cv_path = cv_svg.selectAll("path").data(cv_pie(dataa));
    var cv_text = cv_svg.selectAll("text").data(cv_pie(dataa));

    cv_path.enter()
        .append("path")
        .attr("fill", function (d, i) { return colourList[i]; })
        .attr("d", cv_arc)
        .each(function (d) { this._current = d; });
    cv_text.enter()
        .append("text")
        .attr("transform", function (d) {
            d.innerRadius = 0;
            d.outerRadius = cv_r;
            return "translate(" + cv_arc.centroid(d) + ")";
        })
        .attr("text-anchor", "middle")
        .attr("fill", "#FFFFFF")
        .attr("font-size", "0")
        .text(function (d) { return d.data.key + "(" + d.data.value + ")"; });

    cv_path.transition().duration(500).attrTween("d", cv_arcTween);
    cv_text.transition().duration(500).attr("transform", function (d) {
        d.innerRadius = 0;
        d.outerRadius = cv_r;
        return "translate(" + cv_arc.centroid(d) + ")";
    });

    cv_path.exit().remove();
    cv_text.exit().remove();
}

$(function () {
    $('#emoji-toggle').change(function () {
        $('#console-event').html('Toggle: ' + $(this).prop('checked'))
    })
})

setInterval(function() { UpdateGraph(); }, 2500);



