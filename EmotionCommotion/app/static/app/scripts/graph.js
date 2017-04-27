//*************************************************************************************
// graph.js
// This file updates the pie chart, when it gets the data back from the ajax call
// The graph updates itself every 2 seconds, if data is still the same, 
// i.e. emotion variable has not been changed then the graph will stay the same
//
// The graph relies on D3
//***********************************************************************************



//create the svg and all the elements + classes, created in the #graph div in index.html
var svg = d3.select("#graph").append("svg").append("g")
svg.append("g").attr("class", "slices");
svg.append("g").attr("class", "labelName");
svg.append("g").attr("class", "labelValue");
svg.append("g").attr("class", "lines");

//create height, width and radius of the pie chart
var width = 960;
var height = 450;
var radius = Math.min(width, height) / 2;


//create default pie chart
var pie = d3.layout.pie()
	.sort(null)
	.value(function(d) {
		return d.value;
	});

//set up radius, allow us to create a doughnut pie chart if we make the inner radius smaller
var arc = d3.svg.arc().outerRadius(radius * 0.8).innerRadius(0);
var outerArc = d3.svg.arc().innerRadius(radius * 0.9).outerRadius(radius * 0.9);
var legendRectSize = (radius * 0.05);
var legendSpacing = radius * 0.02;

//set the colours for each part
var div = d3.select("body").append("div").attr("class", "toolTip");
svg.attr("transform", "translate(" + width / 2 + "," + height / 2 + ")");
var color = d3.scale.ordinal().range(["#20c11b", "#f4eb30" , "#1989d3", "#d33917"]);


//this function get new data, by GetEmotion() from microphone which contains
//latest information about emotions, multiple by 100
function newData() {
    result = getEmotion()
    return [
        {label:"Neutral", value: result.neu*100},
        {label:"Happy", value: result.hap*100},
        {label:"Sadness", value: result.sad*100},
        {label:"Angry", value: result.ang*100}
    ];
}

	
//Changes the graph with the new data
//taken from D3 own dynamically updating pie chart example
function change(data) {

	/* ------- PIE SLICES -------*/
	var slice = svg.select(".slices").selectAll("path.slice")
		.data(pie(data), function(d){ return d.data.label });

	slice.enter()
		.insert("path")
		.style("fill", function(d) { return color(d.data.label); })
		.attr("class", "slice");

	slice
		.transition().duration(1000)
		.attrTween("d", function(d) {
			this._current = this._current || d;
			var interpolate = d3.interpolate(this._current, d);
			this._current = interpolate(0);
			return function(t) {
				return arc(interpolate(t));
			};
		})
	slice

	slice.exit()
		.remove();


        //translate using css animations
	var legend = svg.selectAll('.legend')
		.data(color.domain())
		.enter()
		.append('g')
		.attr('class', 'legend')
		.attr('transform', function(d, i) {
			var height = legendRectSize + legendSpacing;
			var offset =  height * color.domain().length / 2;
			var horz = -3 * legendRectSize;
			var vert = i * height - offset;
			return 'translate(' + (horz+335) + ',' + -(vert+150) + ')';
		});

	legend.append('rect')
		.attr('width', legendRectSize)
		.attr('height', legendRectSize)
		.style('fill', color)
		.style('stroke', color);

	legend.append('text')
		.attr('x', legendRectSize + legendSpacing)
		.attr('y', legendRectSize - legendSpacing)
		.text(function(d) { return d; });

	/* ------- TEXT LABELS -------*/

	var text = svg.select(".labelName").selectAll("text")
		.data(pie(data), function(d){ return d.data.label });

	text.enter()
		.append("text")
		.attr("dy", ".35em")
		.text(function(d) {
			return (d.data.label+": "+d.value+"%");
		});

	function midAngle(d){
		return d.startAngle + (d.endAngle - d.startAngle)/2;
	}

	text
		.transition().duration(1000)
		.attrTween("transform", function(d) {
			this._current = this._current || d;
			var interpolate = d3.interpolate(this._current, d);
			this._current = interpolate(0);
			return function(t) {
				var d2 = interpolate(t);
				var pos = outerArc.centroid(d2);
				pos[0] = radius * (midAngle(d2) < Math.PI ? 1 : -1);
				return "translate("+ pos +")";
			};
		})
		.styleTween("text-anchor", function(d){
			this._current = this._current || d;
			var interpolate = d3.interpolate(this._current, d);
			this._current = interpolate(0);
			return function(t) {
				var d2 = interpolate(t);
				return midAngle(d2) < Math.PI ? "start":"end";
			};
		})
		.text(function(d) {
			return (d.data.label+": "+d.value+"%");
		});


	text.exit()
		.remove();

	/* ------- SLICE TO TEXT POLYLINES -------*/

	var polyline = svg.select(".lines").selectAll("polyline")
		.data(pie(data), function(d){ return d.data.label });

	polyline.enter()
		.append("polyline");

	polyline.transition().duration(1000)
		.attrTween("points", function(d){
			this._current = this._current || d;
			var interpolate = d3.interpolate(this._current, d);
			this._current = interpolate(0);
			return function(t) {
				var d2 = interpolate(t);
				var pos = outerArc.centroid(d2);
				pos[0] = radius * 0.95 * (midAngle(d2) < Math.PI ? 1 : -1);
				return [arc.centroid(d2), outerArc.centroid(d2), pos];
			};
		});

	polyline.exit()
		.remove();
};

//uses newData to change the graph
setInterval(function () { change(newData()); }, 2500);
