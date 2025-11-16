// Sentinel-2 Water/Land Classification using NIR
//
// This script creates a binary water/land classification from Sentinel-2 NIR imagery
// with interactive date selection, threshold adjustment, and polygon/rectangle AOI selection.
//
// Instructions:
// 1. Draw a polygon or rectangle on the map to define your area of interest
// 2. Use the two date sliders to select your start and end dates
// 3. Adjust the NIR threshold slider to fine-tune water detection (lower = more water)
// 4. Choose colored or grayscale visualization
// 5. Click "Run Analysis" to generate the classification
// 6. Check the Tasks tab to export to Google Drive

// Cloud masking function
function maskS2clouds(image) {
  var qa = image.select('QA60');

  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

  return image.updateMask(mask).divide(10000);
}

// Function to classify water and land based on NIR threshold
function classifyWaterLand(image, threshold) {
  // Select NIR band (B8)
  var nir = image.select('B8');

  // Create binary classification
  // Water has low NIR reflectance, land has high NIR reflectance
  // 0 = water, 1 = land
  var classified = nir.gt(threshold).rename('classification');

  return classified;
}

// Main analysis function
function runAnalysis(earliest, latest, statusLabel, thresholdSlider, grayscaleCheckbox) {
  // Get the current geometry from drawing tools
  var drawnGeometry = Map.drawingTools().layers().get(0).toGeometry();

  // Check if geometry exists and is not empty
  if (!drawnGeometry) {
    statusLabel.setValue('ERROR: Please draw a polygon on the map first!');
    statusLabel.style().set('color', 'red');
    return;
  }

  // Get threshold value
  var nirThreshold = thresholdSlider.getValue();

  // Update status
  statusLabel.setValue('Processing NIR imagery with threshold: ' + nirThreshold.toFixed(3) + '...');
  statusLabel.style().set('color', 'blue');

  // Clear previous layers
  Map.layers().reset();

  var dataset = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                    .filterDate(earliest, latest)
                    .filterBounds(drawnGeometry)
                    // Pre-filter to get less cloudy granules.
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',5))
                    .map(maskS2clouds);

  // Check if any images were found
  var count = dataset.size().getInfo();
  if (count === 0) {
    statusLabel.setValue('ERROR: No images found for this date range. Try expanding dates.');
    statusLabel.style().set('color', 'red');
    return;
  }

  // Clip the composite images to the specified region.
  var composite = dataset.mean();
  var clippedComposite = composite.clip(drawnGeometry);

  // Perform water/land classification
  var classified = classifyWaterLand(clippedComposite, nirThreshold);

  // Visualization parameters for binary classification
  var useGrayscale = grayscaleCheckbox.getValue();
  var classificationVis = useGrayscale ? {
    min: 0,
    max: 1,
    palette: ['000000', 'ffffff']  // Grayscale: Black for water (0), White for land (1)
  } : {
    min: 0,
    max: 1,
    palette: ['0066ff', '228b22']  // Colored: Blue for water (0), Green for land (1)
  };

  // Add classification to map
  Map.addLayer(classified, classificationVis, 'Water/Land Classification');

  // Also show NIR band for reference
  var nirVis = {
    bands: ['B8'],
    min: 0,
    max: 0.3,
    palette: ['000000', 'ffffff']
  };
  Map.addLayer(clippedComposite, nirVis, 'NIR (B8)', false);  // Hidden by default

  // Calculate water and land areas
  var pixelArea = ee.Image.pixelArea();
  var classifiedArea = classified.multiply(pixelArea);

  var stats = classifiedArea.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: drawnGeometry,
    scale: 10,
    maxPixels: 1e9
  });

  // Get areas
  var landArea = ee.Number(stats.get('classification'));
  var totalArea = pixelArea.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: drawnGeometry,
    scale: 10,
    maxPixels: 1e9
  });
  var waterArea = ee.Number(totalArea.get('area')).subtract(landArea);

  // Convert to hectares
  var waterHa = waterArea.divide(10000);
  var landHa = landArea.divide(10000);

  // Update status with statistics
  var waterPercent = waterArea.divide(totalArea.get('area')).multiply(100);
  var landPercent = landArea.divide(totalArea.get('area')).multiply(100);

  waterHa.evaluate(function(wh) {
    landHa.evaluate(function(lh) {
      waterPercent.evaluate(function(wp) {
        landPercent.evaluate(function(lp) {
          statusLabel.setValue(
            '✓ Classification complete! Check Tasks tab to export.\n' +
            'Water: ' + wh.toFixed(2) + ' ha (' + wp.toFixed(1) + '%)\n' +
            'Land: ' + lh.toFixed(2) + ' ha (' + lp.toFixed(1) + '%)\n' +
            'Images used: ' + count
          );
          statusLabel.style().set('color', 'green');
        });
      });
    });
  });

  // Define the export parameters for binary classification
  var classificationExportParams = {
    image: classified.toByte(),  // Convert to byte to save space
    description: 'WaterLand_Classification',
    scale: 10,
    region: drawnGeometry,
    fileFormat: 'GeoTIFF',
    maxPixels: 1e9
  };

  // Export the binary classification to Google Drive.
  Export.image.toDrive(classificationExportParams);
}

// Create UI Panel
var panel = ui.Panel({
  style: {
    width: '350px',
    position: 'top-left',
    padding: '8px'
  }
});

// Add title
var title = ui.Label({
  value: 'Water/Land Classification (NIR)',
  style: {
    fontSize: '20px',
    fontWeight: 'bold',
    margin: '0 0 10px 0'
  }
});
panel.add(title);

// Add instructions
var instructions = ui.Label({
  value: '1. Draw a polygon/rectangle on map\n2. Set start and end dates\n3. Adjust NIR threshold\n4. Choose visualization style\n5. Click "Run Analysis"',
  style: {
    fontSize: '13px',
    margin: '0 0 10px 0',
    whiteSpace: 'pre'
  }
});
panel.add(instructions);

// Add separator
panel.add(ui.Panel({
  style: {
    height: '1px',
    backgroundColor: 'gray',
    margin: '10px 0'
  }
}));

// Start Date Section
var startDateLabel = ui.Label({
  value: 'Start Date:',
  style: {
    fontSize: '14px',
    fontWeight: 'bold',
    margin: '5px 0 2px 0'
  }
});
panel.add(startDateLabel);

var startDateSlider = ui.DateSlider({
  start: '2020-01-01',
  end: Date.now(),
  value: '2024-05-01',
  period: 1,
  style: {
    stretch: 'horizontal'
  }
});
panel.add(startDateSlider);

// End Date Section
var endDateLabel = ui.Label({
  value: 'End Date:',
  style: {
    fontSize: '14px',
    fontWeight: 'bold',
    margin: '10px 0 2px 0'
  }
});
panel.add(endDateLabel);

var endDateSlider = ui.DateSlider({
  start: '2020-01-01',
  end: Date.now(),
  value: '2024-10-30',
  period: 1,
  style: {
    stretch: 'horizontal'
  }
});
panel.add(endDateSlider);

// Add separator
panel.add(ui.Panel({
  style: {
    height: '1px',
    backgroundColor: 'gray',
    margin: '10px 0'
  }
}));

// NIR Threshold Section
var thresholdLabel = ui.Label({
  value: 'NIR Threshold (Water Detection):',
  style: {
    fontSize: '14px',
    fontWeight: 'bold',
    margin: '5px 0 2px 0'
  }
});
panel.add(thresholdLabel);

// Threshold explanation
var thresholdExplanation = ui.Label({
  value: 'Lower = More Water | Higher = Less Water',
  style: {
    fontSize: '11px',
    color: 'gray',
    margin: '0 0 5px 0',
    fontStyle: 'italic'
  }
});
panel.add(thresholdExplanation);

// Threshold value display
var thresholdValueLabel = ui.Label({
  value: 'Current: 0.150',
  style: {
    fontSize: '12px',
    margin: '0 0 5px 0',
    color: '#2c5aa0'
  }
});
panel.add(thresholdValueLabel);

// Threshold slider
var thresholdSlider = ui.Slider({
  min: 0.05,
  max: 0.30,
  value: 0.15,
  step: 0.01,
  style: {
    stretch: 'horizontal'
  },
  onChange: function(value) {
    thresholdValueLabel.setValue('Current: ' + value.toFixed(3));
  }
});
panel.add(thresholdSlider);

// Add preset buttons
var presetPanel = ui.Panel({
  layout: ui.Panel.Layout.flow('horizontal'),
  style: {
    margin: '5px 0'
  }
});

var presetLabel = ui.Label({
  value: 'Presets:',
  style: {
    fontSize: '11px',
    margin: '2px 5px 0 0'
  }
});
presetPanel.add(presetLabel);

var conservativeButton = ui.Button({
  label: 'Conservative',
  style: {
    fontSize: '10px',
    padding: '2px 4px',
    margin: '0 2px'
  },
  onClick: function() {
    thresholdSlider.setValue(0.10);
  }
});
presetPanel.add(conservativeButton);

var moderateButton = ui.Button({
  label: 'Moderate',
  style: {
    fontSize: '10px',
    padding: '2px 4px',
    margin: '0 2px'
  },
  onClick: function() {
    thresholdSlider.setValue(0.15);
  }
});
presetPanel.add(moderateButton);

var aggressiveButton = ui.Button({
  label: 'Aggressive',
  style: {
    fontSize: '10px',
    padding: '2px 4px',
    margin: '0 2px'
  },
  onClick: function() {
    thresholdSlider.setValue(0.20);
  }
});
presetPanel.add(aggressiveButton);

panel.add(presetPanel);

// Add separator
panel.add(ui.Panel({
  style: {
    height: '1px',
    backgroundColor: 'gray',
    margin: '10px 0'
  }
}));

// Visualization options
var visualizationLabel = ui.Label({
  value: 'Visualization:',
  style: {
    fontSize: '14px',
    fontWeight: 'bold',
    margin: '5px 0 5px 0'
  }
});
panel.add(visualizationLabel);

var grayscaleCheckbox = ui.Checkbox({
  label: 'Grayscale (Black=Water, White=Land)',
  value: false,
  style: {
    fontSize: '11px',
    margin: '2px 0 5px 0'
  }
});
panel.add(grayscaleCheckbox);

var colorNote = ui.Label({
  value: 'Note: Exported GeoTIFF is always binary (0=Water, 1=Land)',
  style: {
    fontSize: '10px',
    color: 'gray',
    fontStyle: 'italic',
    margin: '0 0 5px 0',
    whiteSpace: 'pre-wrap'
  }
});
panel.add(colorNote);

// Status label
var statusLabel = ui.Label({
  value: 'Ready. Draw a polygon and click Run Analysis.',
  style: {
    fontSize: '12px',
    color: 'gray',
    margin: '10px 0',
    whiteSpace: 'pre-wrap'
  }
});
panel.add(statusLabel);

// Run button
var runButton = ui.Button({
  label: '▶ Run Analysis',
  style: {
    stretch: 'horizontal',
    margin: '5px 0'
  },
  onClick: function() {
    var earliest = ee.Date(startDateSlider.getValue()[0]);
    var latest = ee.Date(endDateSlider.getValue()[0]);
    runAnalysis(earliest, latest, statusLabel, thresholdSlider, grayscaleCheckbox);
  }
});
panel.add(runButton);

// Add info section
panel.add(ui.Panel({
  style: {
    height: '1px',
    backgroundColor: 'gray',
    margin: '10px 0'
  }
}));

var infoLabel = ui.Label({
  value: 'Method: NIR threshold classification\nResolution: 10m\nBand: B8 (NIR 835nm)\nCloud coverage: < 5%\nOutput: Binary raster (0=Water, 1=Land)',
  style: {
    fontSize: '11px',
    color: 'gray',
    whiteSpace: 'pre'
  }
});
panel.add(infoLabel);

// Add legend
panel.add(ui.Panel({
  style: {
    height: '1px',
    backgroundColor: 'gray',
    margin: '10px 0'
  }
}));

var legendLabel = ui.Label({
  value: 'Legend (Colored Mode):',
  style: {
    fontSize: '12px',
    fontWeight: 'bold',
    margin: '5px 0'
  }
});
panel.add(legendLabel);

// Water legend
var waterLegend = ui.Panel({
  layout: ui.Panel.Layout.flow('horizontal'),
  style: {
    margin: '2px 0'
  }
});
var waterColor = ui.Label({
  style: {
    backgroundColor: '0066ff',
    padding: '8px',
    margin: '0 5px 0 0'
  }
});
var waterText = ui.Label({
  value: 'Water (0) - Blue',
  style: {
    fontSize: '11px',
    margin: '2px 0'
  }
});
waterLegend.add(waterColor);
waterLegend.add(waterText);
panel.add(waterLegend);

// Land legend
var landLegend = ui.Panel({
  layout: ui.Panel.Layout.flow('horizontal'),
  style: {
    margin: '2px 0'
  }
});
var landColor = ui.Label({
  style: {
    backgroundColor: '228b22',
    padding: '8px',
    margin: '0 5px 0 0'
  }
});
var landText = ui.Label({
  value: 'Land (1) - Green',
  style: {
    fontSize: '11px',
    margin: '2px 0'
  }
});
landLegend.add(landColor);
landLegend.add(landText);
panel.add(landLegend);

// Grayscale legend note
var grayscaleLegendNote = ui.Label({
  value: 'Grayscale Mode: Black=Water (0), White=Land (1)',
  style: {
    fontSize: '10px',
    color: 'gray',
    fontStyle: 'italic',
    margin: '5px 0 0 0'
  }
});
panel.add(grayscaleLegendNote);

// Add panel to map
ui.root.insert(0, panel);

// Set initial map center (Quinhagak, Alaska)
Map.setCenter(-161.90402886481778, 59.750359237437635, 10);

// Add drawing tools (polygon and rectangle)
Map.drawingTools().setShown(true);
Map.drawingTools().setDrawModes(['polygon', 'rectangle']);
