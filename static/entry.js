/// <reference path="jquery.d.ts" />
var WIDTH = 28;
var HEIGHT = 28;
var maxWIDTH = 20;
var maxHEIGHT = 20;
var SCALE = 5;
var submit_timing = 500; // 1000 = 1 second

function main() {
    for (var r = 1; r < 3; r++) {
        for (var i = 0; i < 5; i++) {
            $("<div class=\"col-xs-2\">\n                Number " + (r * 5 + i - 5) + "\n                <canvas id=\"canvas" + (r * 5 + i - 5) + "\" width=" + WIDTH * SCALE + " height=" + WIDTH * SCALE + " style=\"border:1px solid #000000\"></canvas>\n                <div class=\"progress\">\n                    <div id=\"progress" + (r * 5 + i - 5) + "\" class=\"progress-bar\" role=\"progressbar\" style=\"width:0%;\">0%</div>\n                </div>\n              </div>").appendTo("#row" + r);
        }
    }
    create_toolbar();
    auto_submit();
    create_samples();
}
function toGrayScale(d) {
    var gray = new Uint8ClampedArray(d.width * d.height);
    for (var i = 0; i < d.width * d.height * 4; i += 4) {
        var r = d.data[i];
        var g = d.data[i + 1];
        var b = d.data[i + 2];
        var a = d.data[i + 3];
        var c = 0;
        if (r == 0 && g == 0 && b == 0 && a != 0) {
            c = 255;
        }
        else {
            c = 0;
        }
        gray[i >> 2] = c;
    }
    return Array.from(gray);
}
function fromGrayScale(a) {
    //give an array of 784 pixels of 0..255, return ImageData
}
function visualizeExplain(a) {
    // convert an array of 28* 28 number, whose range is -1 to 1,  to an Uint8ClampedArray of 28 * 28 *4 of RGBA ,
    // with -1 convert to red, 1 to green, -0.5 to half red.
    var size = WIDTH * HEIGHT * 4;
    var data = new Uint8ClampedArray(size);
    for (var i = 0; i < size; i += 4) {
        var v = a[i >> 2];
        var r = 0;
        var g = 0;
        var b = 0;
        if (v < -0.5) {
            r = 255 * (-v);
        }
        if (v > 0.5) {
            g = 255 * v;
        }
        if (-0.5 <= v && v <= 0.5) {
            r = 255;
            g = 255;
            b = 255;
        }
        data[i] = r;
        data[i + 1] = g;
        data[i + 2] = b;
        data[i + 3] = 255;
    }
    return data;
}

// finds corner coordinates that defines a rectangle around the pixel mass
// TO DO: refractor using imageData directly instead of grayScale (hint: use mod to get j)
function findPixelBounds(data) {
    
    // image bounds
    var leftX = -1;
    var leftY = -1;
    var rightX = 0;
    var rightY = 0;

    //center of mass
    var totalMass = 0;
    var xMass = 0;
    var yMass = 0;
    var centerMassX;
    var centerMassY;

    //console.log(data);

    for (var i = 0; i < HEIGHT; i++) {

        for (var j = 0; j < WIDTH; j++) {
            
            var pixelMass =  data[i * WIDTH + j];

            totalMass = totalMass + pixelMass;
            xMass = xMass + j * pixelMass;
            yMass = yMass + i * pixelMass;
        
            if (pixelMass != 0) {

                if (leftX == -1 || leftY == -1) {
                    leftX = j;
                    leftY = i;         
                }
                if (j < leftX){
                    leftX = j;
                }
                if (j > rightX){
                    rightX = j;
                }
                rightY = i;
            }
        }
    }

    if (totalMass==0){totalMass = 1;}
    centerMassX = xMass / totalMass;
    centerMassY = yMass / totalMass;
    return [leftX, leftY, rightX, rightY, centerMassX, centerMassY];
}


function showExplain(explain, id) {
    var canvas = document.getElementById('standard_canvas');
    var ctx = canvas.getContext("2d");

    var imageData = new ImageData(visualizeExplain(explain), WIDTH, HEIGHT);
    ctx.putImageData(imageData, 0, 0);
    
    var ctx1 = document.getElementById(id).getContext("2d");
    ctx1.drawImage(canvas, 0, 0, WIDTH * SCALE, HEIGHT * SCALE);
    ctx1.drawImage(document.getElementById('sketch_transform'), 0, 0, WIDTH * SCALE, HEIGHT * SCALE);
}

function submit() {
    // convert the data in canvas #sketch to grey scale 28 * 28 uint8 byte array
    // send it to server.
    // get post result and render it in the 10 canvas below.
    var canvas = document.getElementById('standard_canvas');
    var canvas = document.getElementById('sketch_transform');
    var ctx = canvas.getContext("2d");

    ctx.clearRect(0, 0, WIDTH, HEIGHT);
    var sketch = document.getElementById('sketch');
    ctx.drawImage(sketch, 0, 0, WIDTH, HEIGHT);
    var scaleWidth = Math.ceil(sketch.width / WIDTH);
    var scaleHeight = Math.ceil(sketch.height / HEIGHT);

    var imageData = ctx.getImageData(0, 0, WIDTH, HEIGHT);
    var d = toGrayScale(imageData);

    
    //console.log(imageData);
    var [leftX, leftY, rightX, rightY, centerMassX, centerMassY] = findPixelBounds(d); // gets rectangle dims and centroid
    
    ctx.clearRect(0, 0, WIDTH, HEIGHT);
    
    if ($('input#check_center').is(':checked')) {  
        
        var min_dx = (WIDTH - maxWIDTH) / 2;
        var min_dy = (HEIGHT - maxHEIGHT) / 2;
        var sWidth = (rightX - leftX + 1) * scaleWidth;
        var sHeight = (rightY - leftY + 1) * scaleHeight;
        var sx = leftX * scaleWidth;
        var sy = leftY * scaleHeight;
        
        // Preserve Height to Weight ratio in output image keeping Height constant
        var dWidth = (sWidth * maxHEIGHT) / sHeight;
        if (dWidth > maxWIDTH){dWidth = maxWIDTH;}
        // Shift dx based on dWidth
        var dx = min_dx + (maxWIDTH - dWidth) / 2;
        
        // shift based on center of mass
        var dy;
        dy = min_dy + (sHeight / 2 - (centerMassY - leftY) * scaleHeight) * (maxHEIGHT / sHeight);
        if (dy < 0){ dy= 0;}


        // redraw and transform image
        ctx.drawImage(sketch, sx, sy, sWidth, sHeight, dx, dy, dWidth, maxHEIGHT);
    }

    else {
        ctx.drawImage(sketch, 0, 0, WIDTH, HEIGHT);
    }

    var imageData = ctx.getImageData(0, 0, WIDTH, HEIGHT);
    var d = toGrayScale(imageData);

    var data = JSON.stringify({
        'procedure': $('#model').val(),
        "user_input": d
    });
    $.post('../handle_drawing', data, function (data1, textstatus, jqXHR) {
        for (var i = 0; i < 10; i++) {
            showExplain(data1.explains[i], 'canvas' + i);
            var score = data1.scores[i] + '%';
            $('#progress' + i).css('width', score).text(score);
        }
    }, 'json');
}
function erase() {
    var canvas = document.getElementById('sketch');
    var ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    $('#sketch').sketch('actions', []);
}
function create_toolbar() {
    $('.colorbtn').on('click', function () {
        $('.colorbtn').css('borderWidth', '1px');
        $(this).css('borderWidth', '3px');
    });
}
function auto_submit() {
    var id = 0;
    $('#sketch').mousedown(function () {
        id = setInterval(submit, submit_timing);
    });
    $('#sketch').on('mouseup mouseleave', function () {
        window.clearInterval(id);
    });
}
function create_samples() {
    for (var i = 0; i < 10; i++) {
        $("<img id=\"sample" + i + "\" class=\"sample_images\" width=" + WIDTH * 2 + " height=" + HEIGHT * 2 + " src=\"" + i + ".png\" style=\"border:1px solid #000000\" onclick=\"handle_image_click(this)\"></img>")
            .appendTo("#samples");
    }
}
function drawImage(id, a) {
    var canvas = document.getElementById(id);
    var ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(a, 0, 0, canvas.width, canvas.height);
    return ctx;
}
function handle_image_click(a) {
    erase();
    $('#sketch').sketch('actions').push({
        tool: 'blit',
        src: a
    });
    $('#sketch').sketch().redraw();
    submit();
}
$(document).on('change', '.form-control', function(){
    var target = $(this).data('target');
    var show = $("option:selected", this).data('show');
    $(target).children().addClass('hide');
    $(show).removeClass('hide');
});
$(document).ready(function(){
    $('.form-control').trigger('change');
});
window.retrieve = function () {
    $.get('../mnist_pics', '', function (data, textStatus, jqXHR) {
        $('img.sample_images').attr("src", function (i, old_attr) {
            var d = new Date();
            return i + ".png?" + d.getTime();
        });
    });
};
$(document).ready(function() {
    retrieve();
});
main();
