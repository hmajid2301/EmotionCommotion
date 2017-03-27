var svg = d3.select("#graph")
	.append("svg")
	.append("g")

svg.append("g")
	.attr("class", "slices");
svg.append("g")
	.attr("class", "labelName");
svg.append("g")
	.attr("class", "labelValue");
svg.append("g")
	.attr("class", "lines");

var width = 960,
	height = 450,
	radius = Math.min(width, height) / 2;

var pie = d3.layout.pie()
	.sort(null)
	.value(function(d) {
		return d.value;
	});

var arc = d3.svg.arc()
	.outerRadius(radius * 0.8)
	.innerRadius(0);

var outerArc = d3.svg.arc()
	.innerRadius(radius * 0.9)
	.outerRadius(radius * 0.9);

var legendRectSize = (radius * 0.05);
var legendSpacing = radius * 0.02;


var div = d3.select("body").append("div").attr("class", "toolTip");

svg.attr("transform", "translate(" + width / 2 + "," + height / 2 + ")");

var color = d3.scale.ordinal()
	.range(["#20c11b", "#f4eb30" , "#1989d3", "#d33917"]);


//function newData() {
//	result = GetEmotion()
//	console.log(result.neu)
//	return [
//			{label:"Neutral", value: result.neu*100}, 
//			{label:"Happy", value: result.hap*100}, 
//			{label:"Sadness", value: result.sad*100},
//			{label:"Angry", value: result.ang*100}
//	];
//}

function newData() {

	ran = [Math.floor((Math.random() * 10) + 1),
		   Math.floor((Math.random() * 10) + 1),
		   Math.floor((Math.random() * 10) + 1),
		   Math.floor((Math.random() * 10) + 1)]

	sum = ran.reduce(function (a, b) { return a + b; }, 0);


	return [
			{ label: "Neutral", value: parseInt((ran[0] / sum) * 100) },
			{ label: "Happy", value: parseInt((ran[1] / sum) * 100) },
			{ label: "Sadness", value: parseInt((ran[2] / sum) * 100) },
			{ label: "Angry", value: parseInt((ran[3] / sum) * 100) }
	];
}
	
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

function randombetween(min, max) {
	return Math.floor(Math.random() * (max - min + 1) + min);
}

(function ($) {
	$.each(['show', 'hide'], function (i, ev) {
		var el = $.fn[ev];
		$.fn[ev] = function () {
			this.trigger(ev);
			return el.apply(this, arguments);
		};
	});
})(jQuery);

function update() {
	console.log("HELLO!");
}

//$('#graph').on('show', change(newData()));
$('#graph').on('show', update);
setInterval(function () { change(newData()); }, 2500);
