var images = [];
var parameterImages = [];

function loadIdx(filename, imageArray, callback) {
  var req = new XMLHttpRequest();
  req.onload = function() {
    populateArray(req.response, imageArray);
    callback();
  }
  req.responseType = "arraybuffer";
  req.open("GET", filename, true);
  req.send();
}

function populateArray(rawIdx, imageArray) {
  var idx = new Uint8Array(rawIdx);
  var numDimensions = idx[3];
  var dimensions = [];
  for (var i = 0; i < numDimensions; i++) {
    var dimension = intFromBigEndianByteArray(idx.subarray(4 + 4*i, 8 + 4*i));
    dimensions.push(dimension);
  }
  constructImages(idx, dimensions, imageArray);
}

function constructImages(idx, dimensions, imageArray) {
  var counter = 4 + 4*dimensions.length;
  for (var i = 0; i < dimensions[0]; i++) {
    var image = [];
    for (var j = 0; j < dimensions[1]; j++) {
      var row = [];
      for (var k = 0; k < dimensions[2]; k++) {
        row.push(idx[counter++]);
      }
      image.push(row);
    }
    imageArray.push(image);
  }
}

loadIdx("parameters", parameterImages, function () {
  var parameterImagesDiv = document.createElement("div");
  createParameterImageViewers(parameterImagesDiv);
  document.body.appendChild(parameterImagesDiv);
});

loadIdx("test-images", images, function () {
  var imageBrowserDiv = document.createElement("div");
  var incorrectPredictionsDiv = document.createElement("div");
  createDynamicViewer(imageBrowserDiv, images, 0);
  createIncorrectPredictionViewers(incorrectPredictionsDiv);
  document.body.appendChild(imageBrowserDiv);
  document.body.appendChild(incorrectPredictionsDiv);
});

function createParameterImageViewers(div) {
  for (var i = 0; i < parameterImages.length; i++) {
    var subtitle = createSubtitle(i);
    var image = parameterImages[i];
    createStaticViewer(div, image, subtitle);
  }
}

function createIncorrectPredictionViewers(div) {
  var req = new XMLHttpRequest();
  req.onload = function() {
    createStaticComparisons(div, JSON.parse(req.responseText));
  }
  req.open("GET", "incorrect_predictions", true);
  req.send();
}

function createStaticComparisons(div, incorrectPredictions) {
  var counter = 0;
  var createSomeViewers = function() {
    for (var i = 0; i < 10; i++) {
      var prediction = incorrectPredictions[counter];
      var subtitle = createSubtitle(
        "predicted: " + prediction[2] + "<br />actual:    " + prediction[1]
      );
      subtitle.style.whiteSpace = "pre";
      var image = images[prediction[0]];
      createStaticViewer(div, image, subtitle);
      if (++counter == incorrectPredictions.length) { return }
    }
    setTimeout(createSomeViewers, 100);
  }
  createSomeViewers();
}

function createDynamicViewer(div, imageArray, startingIndex) {
  var image = imageArray[startingIndex];
  var table = createTable(image);
  var span = document.createElement("span");
  span.className = "container";
  span.appendChild(table);
  var input = createInputForTable(table, imageArray);
  input.value = startingIndex;
  span.appendChild(input);
  input.focus();
  div.appendChild(span);
}

function createStaticViewer(div, image, subtitle) {
  var table = createTable(image);
  var span = document.createElement("span");
  span.className = "container";
  span.appendChild(table);
  span.appendChild(subtitle);
  div.appendChild(span);
}

function createTable(image) {
  var table = document.createElement("table");
  var numRows = image.length;
  var numCols = image[0].length;
  for (var i = 0; i < numRows; i++) {
    var tr = table.insertRow();
    for (var j = 0; j < numCols; j++) {
      tr.insertCell();
    }
  }
  showImage(image, table);
  return table;
}

function createSubtitle(content) {
  var subtitle = document.createElement("span");
  subtitle.innerHTML = content;
  subtitle.className = "subtitle";
  return subtitle;
}

function createInputForTable(table, imageArray) {
  var input = document.createElement("input");
  input.type = "number";
  input.onchange = function() {
    if (input.value < 0) {
      input.value = 0;
    } else if (input.value >= imageArray.length) {
      input.value = imageArray.length - 1;
    } else {
      showImage(imageArray[input.value], table);
    }
  }
  return input;
}

function showImage(image, table) {
  var numRows = table.rows.length;
  var numCols = table.rows[0].cells.length;
  for (var i = 0; i < numRows; i++) {
    var tr = table.rows[i];
    var row = image[i];
    for (var j = 0; j < numCols; j++) {
      var cellString = Number(255 - image[i][j]).toString(16);
      if (cellString.length == 1) { cellString = "0" + cellString; }
      tr.cells[j].style.backgroundColor = "#" + cellString + cellString + cellString;
    }
  }
}

function intFromBigEndianByteArray(bytes) {
  return bytes[0]*Math.pow(2,24) + bytes[1]*Math.pow(2,16) + bytes[2]*Math.pow(2,8) + bytes[3];
}

