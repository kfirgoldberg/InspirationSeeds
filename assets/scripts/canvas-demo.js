// Interactive Canvas Demo for Inspiration Seeds
(function() {
    'use strict';

    const IMAGE_BASE_PATH = 'static/figures/demo-images/';

    // Available combinations: key is sorted pair, value is result filename
    const COMBINATIONS = {
        'arch1_food1': 'arch1_food1_result_2.jpg',
        'fashion5_food1': 'fashion5_food1_result_8.jpg',
        'food1_nature2': 'food1_nature2_result_2.jpg',
        'food1_sea4': 'food1_sea4_result_3.jpg'
    };

    // Arrow colors for each combination
    const ARROW_COLORS = {
        'food1_nature2': '#A8DCF0',
        'fashion5_food1': '#888888',
        'arch1_food1': '#70AD47',
        'food1_sea4': '#ED7D31'
    };

    // All available input images
    const INPUT_IMAGES = ['food1', 'nature2', 'fashion5', 'arch1', 'sea4'];

    // Canvas state
    let canvas, ctx;
    let canvasImages = [];
    let loadedImages = {};
    let draggingImage = null;
    let dragOffset = { x: 0, y: 0 };
    let wasDragging = false; // Track if a drag occurred (to prevent click after drag)
    let selectedImage = null; // Currently selected image name (from selector or canvas)
    let selectedCanvasImage = null; // Reference to selected canvas image object (for highlight)
    let highlightedCanvasImages = []; // Images highlighted for compatibility

    // Zoom state
    let zoomLevel = 1.0;
    const ZOOM_MIN = 0.5;
    const ZOOM_MAX = 2.0;
    const ZOOM_STEP = 0.2;
    let initialCanvasState = []; // Store initial images for reset

    // Pan state
    let panOffset = { x: 0, y: 0 };
    let isPanning = false;
    let panStart = { x: 0, y: 0 };

    // Get combination key (sorted alphabetically)
    function getCombinationKey(img1, img2) {
        return [img1, img2].sort().join('_');
    }

    // Check if two images have a combination
    function hasCombination(img1, img2) {
        const key = getCombinationKey(img1, img2);
        return COMBINATIONS.hasOwnProperty(key);
    }

    // Get result image for a combination
    function getResultImage(img1, img2) {
        const key = getCombinationKey(img1, img2);
        return COMBINATIONS[key] || null;
    }

    // Get arrow color for a combination
    function getArrowColor(img1, img2) {
        const key = getCombinationKey(img1, img2);
        return ARROW_COLORS[key] || '#888888';
    }

    // Load an image and cache it
    function loadImage(name) {
        return new Promise((resolve, reject) => {
            if (loadedImages[name]) {
                resolve(loadedImages[name]);
                return;
            }
            const img = new Image();
            img.onload = () => {
                loadedImages[name] = img;
                resolve(img);
            };
            img.onerror = reject;
            // Handle both simple names and full filenames
            const filename = name.includes('.') ? name : name + '.jpg';
            img.src = IMAGE_BASE_PATH + filename;
        });
    }

    // Calculate display dimensions preserving aspect ratio
    function getDisplayDimensions(img, maxWidth, maxHeight) {
        const ratio = Math.min(maxWidth / img.width, maxHeight / img.height);
        return {
            width: img.width * ratio,
            height: img.height * ratio
        };
    }

    // Add an image to the canvas
    function addCanvasImage(name, x, y, maxWidth, maxHeight, isResult = false) {
        const img = loadedImages[name];
        if (!img) return null;

        const dims = getDisplayDimensions(img, maxWidth, maxHeight);
        const canvasImg = {
            name: name,
            img: img,
            x: x,
            y: y,
            width: dims.width,
            height: dims.height,
            isResult: isResult
        };
        canvasImages.push(canvasImg);
        return canvasImg;
    }

    // Find image at position
    function getImageAtPosition(x, y) {
        // Search in reverse order (top images first)
        for (let i = canvasImages.length - 1; i >= 0; i--) {
            const img = canvasImages[i];
            if (x >= img.x && x <= img.x + img.width &&
                y >= img.y && y <= img.y + img.height) {
                return img;
            }
        }
        return null;
    }

    // Get center point of an image
    function getImageCenter(img) {
        return {
            x: img.x + img.width / 2,
            y: img.y + img.height / 2
        };
    }

    // Get edge point of image closest to target
    function getEdgePoint(img, targetX, targetY) {
        const center = getImageCenter(img);
        const dx = targetX - center.x;
        const dy = targetY - center.y;
        const angle = Math.atan2(dy, dx);

        // Calculate intersection with image rectangle
        const halfW = img.width / 2;
        const halfH = img.height / 2;

        let edgeX, edgeY;

        // Check which edge we hit
        const tanAngle = Math.abs(Math.tan(angle));
        if (tanAngle < halfH / halfW) {
            // Hit left or right edge
            edgeX = center.x + (dx > 0 ? halfW : -halfW);
            edgeY = center.y + (dx > 0 ? halfW : -halfW) * Math.tan(angle);
        } else {
            // Hit top or bottom edge
            edgeY = center.y + (dy > 0 ? halfH : -halfH);
            edgeX = center.x + (dy > 0 ? halfH : -halfH) / Math.tan(angle);
        }

        return { x: edgeX, y: edgeY };
    }

    // Draw a straight arrow between two points
    function drawArrow(fromX, fromY, toX, toY, color) {
        ctx.save();
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = 4;

        const dx = toX - fromX;
        const dy = toY - fromY;
        const angle = Math.atan2(dy, dx);

        // Draw straight line
        ctx.beginPath();
        ctx.moveTo(fromX, fromY);
        ctx.lineTo(toX, toY);
        ctx.stroke();

        // Draw arrowhead
        const arrowLen = 12;
        const arrowAngle = Math.PI / 6;

        ctx.beginPath();
        ctx.moveTo(toX, toY);
        ctx.lineTo(
            toX - arrowLen * Math.cos(angle - arrowAngle),
            toY - arrowLen * Math.sin(angle - arrowAngle)
        );
        ctx.lineTo(
            toX - arrowLen * Math.cos(angle + arrowAngle),
            toY - arrowLen * Math.sin(angle + arrowAngle)
        );
        ctx.closePath();
        ctx.fill();

        ctx.restore();
    }

    // Find all active combinations on the canvas
    // Ensures food1 is always the first component (arrows go from food1 → other → result)
    function findActiveCombinations() {
        const combinations = [];
        const inputImages = canvasImages.filter(img => !img.isResult);
        const resultImages = canvasImages.filter(img => img.isResult);

        for (let i = 0; i < inputImages.length; i++) {
            for (let j = i + 1; j < inputImages.length; j++) {
                const img1 = inputImages[i];
                const img2 = inputImages[j];
                const resultFilename = getResultImage(img1.name, img2.name);

                if (resultFilename) {
                    const resultOnCanvas = resultImages.find(r => r.name === resultFilename);
                    if (resultOnCanvas) {
                        // Ensure food1 is always comp1 (first in arrow chain)
                        let comp1 = img1;
                        let comp2 = img2;
                        if (img2.name === 'food1') {
                            comp1 = img2;
                            comp2 = img1;
                        }
                        combinations.push({
                            comp1: comp1,
                            comp2: comp2,
                            result: resultOnCanvas,
                            color: getArrowColor(img1.name, img2.name)
                        });
                    }
                }
            }
        }
        return combinations;
    }

    // Draw all arrows for active combinations
    function drawAllArrows() {
        const combos = findActiveCombinations();

        combos.forEach(combo => {
            const center1 = getImageCenter(combo.comp1);
            const center2 = getImageCenter(combo.comp2);
            const centerResult = getImageCenter(combo.result);

            // Arrow from comp1 to comp2
            const edge1to2 = getEdgePoint(combo.comp1, center2.x, center2.y);
            const edge2from1 = getEdgePoint(combo.comp2, center1.x, center1.y);
            drawArrow(edge1to2.x, edge1to2.y, edge2from1.x, edge2from1.y, combo.color);

            // Arrow from comp2 to result
            const edge2toR = getEdgePoint(combo.comp2, centerResult.x, centerResult.y);
            const edgeRfrom2 = getEdgePoint(combo.result, center2.x, center2.y);
            drawArrow(edge2toR.x, edge2toR.y, edgeRfrom2.x, edgeRfrom2.y, combo.color);
        });
    }

    // Render the canvas
    function render() {
        // Clear canvas with transparent background
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Apply zoom and pan transform
        ctx.save();
        const centerX = canvas.width / (2 * (window.devicePixelRatio || 1));
        const centerY = canvas.height / (2 * (window.devicePixelRatio || 1));
        ctx.translate(centerX, centerY);
        ctx.translate(panOffset.x, panOffset.y);
        ctx.scale(zoomLevel, zoomLevel);
        ctx.translate(-centerX, -centerY);

        // Draw arrows first (behind images)
        drawAllArrows();

        // Draw all images with optional highlight
        canvasImages.forEach(img => {
            // Draw selected border if this is the selected canvas image
            if (selectedCanvasImage && img === selectedCanvasImage) {
                ctx.save();
                ctx.strokeStyle = '#353535';
                ctx.lineWidth = 4;
                ctx.shadowColor = 'rgba(53, 53, 53, 0.4)';
                ctx.shadowBlur = 10;
                ctx.strokeRect(img.x - 2, img.y - 2, img.width + 4, img.height + 4);
                ctx.restore();
            }
            // Draw compatible highlight border if this image is highlighted
            else if (highlightedCanvasImages.includes(img.name)) {
                ctx.save();
                ctx.strokeStyle = '#70AD47';
                ctx.lineWidth = 4;
                ctx.shadowColor = 'rgba(112, 173, 71, 0.5)';
                ctx.shadowBlur = 10;
                ctx.strokeRect(img.x - 2, img.y - 2, img.width + 4, img.height + 4);
                ctx.restore();
            }
            ctx.drawImage(img.img, img.x, img.y, img.width, img.height);
        });

        ctx.restore(); // Restore from zoom transform
    }

    // Mouse/touch event handlers
    function getEventPosition(e) {
        const rect = canvas.getBoundingClientRect();

        let clientX, clientY;
        if (e.touches && e.touches.length > 0) {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
        } else if (e.changedTouches && e.changedTouches.length > 0) {
            clientX = e.changedTouches[0].clientX;
            clientY = e.changedTouches[0].clientY;
        } else {
            clientX = e.clientX;
            clientY = e.clientY;
        }

        // Get position in canvas CSS pixel coordinates
        const displayX = clientX - rect.left;
        const displayY = clientY - rect.top;

        // Transform to account for zoom and pan (inverse of render transform)
        const centerX = rect.width / 2;
        const centerY = rect.height / 2;

        const x = (displayX - centerX - panOffset.x) / zoomLevel + centerX;
        const y = (displayY - centerY - panOffset.y) / zoomLevel + centerY;

        return { x: x, y: y };
    }

    function onMouseDown(e) {
        const pos = getEventPosition(e);
        const img = getImageAtPosition(pos.x, pos.y);

        wasDragging = false; // Reset drag flag

        if (img) {
            draggingImage = img;
            dragOffset.x = pos.x - img.x;
            dragOffset.y = pos.y - img.y;

            // Move to top
            const idx = canvasImages.indexOf(img);
            if (idx > -1) {
                canvasImages.splice(idx, 1);
                canvasImages.push(img);
            }

            canvas.style.cursor = 'grabbing';
            e.preventDefault();
        } else {
            // Start panning on empty space
            isPanning = true;
            const rect = canvas.getBoundingClientRect();
            const clientX = e.touches ? e.touches[0].clientX : e.clientX;
            const clientY = e.touches ? e.touches[0].clientY : e.clientY;
            panStart.x = clientX - rect.left - panOffset.x;
            panStart.y = clientY - rect.top - panOffset.y;
            canvas.style.cursor = 'grabbing';
            e.preventDefault();
        }
    }

    function onMouseMove(e) {
        if (isPanning) {
            // Update pan offset
            const rect = canvas.getBoundingClientRect();
            const clientX = e.touches ? e.touches[0].clientX : e.clientX;
            const clientY = e.touches ? e.touches[0].clientY : e.clientY;
            panOffset.x = clientX - rect.left - panStart.x;
            panOffset.y = clientY - rect.top - panStart.y;
            wasDragging = true;
            render();
            e.preventDefault();
            return;
        }

        const pos = getEventPosition(e);

        if (draggingImage) {
            wasDragging = true; // Mark that a drag occurred
            draggingImage.x = pos.x - dragOffset.x;
            draggingImage.y = pos.y - dragOffset.y;
            render();
            e.preventDefault();
        } else {
            // Update cursor
            const img = getImageAtPosition(pos.x, pos.y);
            canvas.style.cursor = img ? 'grab' : 'default';
        }
    }

    function onMouseUp(e) {
        if (isPanning) {
            isPanning = false;
            canvas.style.cursor = 'default';
        }
        if (draggingImage) {
            canvas.style.cursor = 'grab';
            draggingImage = null;
        }
    }

    // Clear all selection highlights
    function clearSelectionState() {
        selectedImage = null;
        selectedCanvasImage = null;
        highlightedCanvasImages = [];

        const selectorContainer = document.getElementById('image-selector');
        const selectorImages = selectorContainer.querySelectorAll('.selector-image');
        selectorImages.forEach(img => {
            img.classList.remove('selected', 'compatible', 'incompatible');
        });

        render();
    }

    // Update selector to only show images not on canvas
    function updateSelectorVisibility(animate) {
        const onCanvasNames = canvasImages
            .filter(img => !img.isResult)
            .map(img => img.name);
        const selectorImages = document.querySelectorAll('.selector-image');
        selectorImages.forEach(img => {
            const name = img.dataset.imageName;
            const shouldHide = onCanvasNames.includes(name);
            if (shouldHide && img.style.visibility !== 'hidden') {
                if (animate) {
                    img.style.opacity = '0';
                    img.style.transform = 'scale(0.5)';
                    setTimeout(() => {
                        img.style.visibility = 'hidden';
                        img.style.pointerEvents = 'none';
                        img.style.order = '1';
                    }, 300);
                } else {
                    img.style.transition = 'none';
                    img.style.opacity = '0';
                    img.style.visibility = 'hidden';
                    img.style.pointerEvents = 'none';
                    img.style.order = '1';
                    img.offsetHeight;
                    img.style.transition = '';
                }
            } else if (!shouldHide && img.style.visibility === 'hidden') {
                img.style.transition = 'none';
                img.style.visibility = '';
                img.style.opacity = '';
                img.style.transform = '';
                img.style.pointerEvents = '';
                img.style.order = '';
                img.offsetHeight;
                img.style.transition = '';
            }
        });
    }

    // Show compatible images for a given image name
    function showCompatibleImages(imageName, isFromCanvas = false) {
        selectedImage = imageName;
        highlightedCanvasImages = [];

        const selectorContainer = document.getElementById('image-selector');
        const selectorImages = selectorContainer.querySelectorAll('.selector-image');

        // Highlight selector images
        selectorImages.forEach(img => {
            const otherName = img.dataset.imageName;
            if (otherName === imageName) {
                img.classList.add('selected');
                img.classList.remove('compatible', 'incompatible');
            } else if (hasCombination(imageName, otherName)) {
                img.classList.add('compatible');
                img.classList.remove('selected', 'incompatible');
            } else {
                img.classList.add('incompatible');
                img.classList.remove('selected', 'compatible');
            }
        });

        // Highlight compatible images on the canvas
        canvasImages.forEach(canvasImg => {
            if (!canvasImg.isResult && canvasImg.name !== imageName) {
                if (hasCombination(imageName, canvasImg.name)) {
                    highlightedCanvasImages.push(canvasImg.name);
                }
            }
        });

        render();
    }

    // Image selector click handler - single click adds combination with food1
    function onSelectorImageClick(e) {
        const imageName = e.currentTarget.dataset.imageName;
        const resultName = getResultImage('food1', imageName);
        if (!resultName) return;

        const w = parseFloat(canvas.style.width);
        const h = parseFloat(canvas.style.height);
        const imgSize = Math.min(w, h) * 0.28;

        // Load new images, then add at FULL_LAYOUT positions and animate everything
        const toLoad = [imageName, resultName].filter(n => !loadedImages[n]);
        Promise.all(toLoad.map(n => loadImage(n))).then(() => {
            [imageName, resultName].forEach(name => {
                if (!canvasImages.find(img => img.name === name)) {
                    const entry = FULL_LAYOUT.find(e => e.name === name);
                    if (entry) {
                        addCanvasImage(name, entry.xPct * w, entry.yPct * h, imgSize, imgSize, entry.isResult);
                    }
                }
            });

            updateSelectorVisibility(true);
            const targetZoom = (zoomLevel === 1.0) ? Math.max(ZOOM_MIN, zoomLevel - ZOOM_STEP) : zoomLevel;
            animateTransition(targetZoom, 60);
        });
        clearSelectionState();
    }

    // Handle click on canvas image
    function onCanvasClick(e) {
        // Don't handle if we were dragging
        if (wasDragging) {
            wasDragging = false;
            return;
        }

        const pos = getEventPosition(e);
        const clickedImg = getImageAtPosition(pos.x, pos.y);

        if (!clickedImg || clickedImg.isResult) {
            // Clicked empty space or a result image - clear selection
            if (selectedImage) {
                clearSelectionState();
            }
            return;
        }

        // Clicked on an input image on the canvas
        const imageName = clickedImg.name;

        if (selectedImage === imageName) {
            // Deselect
            clearSelectionState();
            return;
        }

        if (selectedImage && highlightedCanvasImages.includes(imageName)) {
            // Second click on compatible image - create combination
            if (hasCombination(selectedImage, imageName)) {
                addCombinationToCanvas(selectedImage, imageName);
            }
            clearSelectionState();
        } else {
            // First click on canvas image - show its compatible pairs
            selectedCanvasImage = clickedImg;
            showCompatibleImages(imageName, true);
        }
    }

    // Add a combination to the canvas
    function addCombinationToCanvas(img1Name, img2Name) {
        const resultFilename = getResultImage(img1Name, img2Name);
        if (!resultFilename) return;

        // Check what's already on canvas
        const existingNames = canvasImages.map(img => img.name);
        const img1OnCanvas = existingNames.includes(img1Name);
        const img2OnCanvas = existingNames.includes(img2Name);
        const resultOnCanvas = existingNames.includes(resultFilename);

        // Get current canvas display dimensions
        const canvasDisplayWidth = parseFloat(canvas.style.width);
        const canvasDisplayHeight = parseFloat(canvas.style.height);
        const imgSize = Math.min(canvasDisplayWidth, canvasDisplayHeight) * 0.28;
        const paddingPct = 0.03; // 3% padding
        const padding = canvasDisplayWidth * paddingPct;

        // Calculate positions for new images
        let nextX = padding;
        let nextY = padding;

        // Find a good position for new images
        if (canvasImages.length > 0) {
            const maxX = Math.max(...canvasImages.map(img => img.x + img.width));
            const maxY = Math.max(...canvasImages.map(img => img.y + img.height));

            if (maxX + imgSize + padding < canvasDisplayWidth) {
                nextX = maxX + padding;
                nextY = padding;
            } else {
                nextX = padding;
                nextY = maxY + padding;
            }
        }

        // Load and add images that aren't on canvas
        const imagesToLoad = [];
        if (!img1OnCanvas) imagesToLoad.push(img1Name);
        if (!img2OnCanvas) imagesToLoad.push(img2Name);
        if (!resultOnCanvas) imagesToLoad.push(resultFilename);

        return Promise.all(imagesToLoad.map(name => loadImage(name))).then(() => {
            imagesToLoad.forEach((name, index) => {
                const isResult = name === resultFilename;
                const xPos = nextX + index * (imgSize + padding);
                const yPos = nextY;
                addCanvasImage(name, xPos, yPos, imgSize, imgSize, isResult);
            });
            render();
        });
    }

    // Zoom functions
    function zoomIn() {
        if (zoomLevel < ZOOM_MAX) {
            zoomLevel = Math.min(ZOOM_MAX, zoomLevel + ZOOM_STEP);
            render();
        }
    }

    function zoomOut() {
        if (zoomLevel > ZOOM_MIN) {
            zoomLevel = Math.max(ZOOM_MIN, zoomLevel - ZOOM_STEP);
            render();
        }
    }

    function animateTransition(targetZoom, targetPanX) {
        const w = parseFloat(canvas.style.width);
        const h = parseFloat(canvas.style.height);
        const duration = 400;
        const startTime = performance.now();
        const startZoom = zoomLevel;
        const startPanX = panOffset.x;
        if (targetPanX === undefined) targetPanX = startPanX;

        // Capture start positions and compute targets from FULL_LAYOUT
        const targets = canvasImages.map(img => {
            const entry = FULL_LAYOUT.find(e => e.name === img.name);
            return {
                startX: img.x,
                startY: img.y,
                endX: entry ? entry.xPct * w : img.x,
                endY: entry ? entry.yPct * h : img.y
            };
        });

        function step(now) {
            const t = Math.min((now - startTime) / duration, 1);
            const ease = t * (2 - t); // ease-out quadratic
            zoomLevel = startZoom + (targetZoom - startZoom) * ease;
            panOffset.x = startPanX + (targetPanX - startPanX) * ease;
            canvasImages.forEach((img, i) => {
                img.x = targets[i].startX + (targets[i].endX - targets[i].startX) * ease;
                img.y = targets[i].startY + (targets[i].endY - targets[i].startY) * ease;
            });
            render();
            if (t < 1) requestAnimationFrame(step);
        }
        requestAnimationFrame(step);
    }

    function resetView() {
        // Reset zoom and pan
        zoomLevel = 1.0;
        panOffset = { x: 0, y: 0 };

        // Get current canvas dimensions
        const canvasDisplayWidth = parseFloat(canvas.style.width);
        const canvasDisplayHeight = parseFloat(canvas.style.height);
        const imgSize = Math.min(canvasDisplayWidth, canvasDisplayHeight) * 0.28;

        // Restore initial canvas state with scaled positions
        canvasImages = initialCanvasState.map(img => {
            const dims = getDisplayDimensions(img.img, imgSize, imgSize);
            return {
                ...img,
                x: img.xPct * canvasDisplayWidth,
                y: img.yPct * canvasDisplayHeight,
                width: dims.width,
                height: dims.height
            };
        });

        // Clear selection state and update selector
        clearSelectionState();
        updateSelectorVisibility();
    }

    // Initial layout positions as percentages of canvas dimensions
    // These are relative to a reference canvas size and will be scaled
    const INITIAL_LAYOUT = [
        { name: 'food1', xPct: 0.17, yPct: 0.48, isResult: false },
        { name: 'nature2', xPct: 0.41, yPct: 0.23, isResult: false },
        { name: 'fashion5', xPct: 0.45, yPct: 0.65, isResult: false },
        { name: 'food1_nature2_result_2.jpg', xPct: 0.09, yPct: 0.07, isResult: true },
        { name: 'fashion5_food1_result_8.jpg', xPct: 0.70, yPct: 0.39, isResult: true }
    ];

    // Full layout after all selector images have been added
    const FULL_LAYOUT = [
        { name: 'nature2', xPct: 0.48, yPct: 0.17, isResult: false },
        { name: 'food1_nature2_result_2.jpg', xPct: 0.17, yPct: -0.01, isResult: true },
        { name: 'fashion5', xPct: 0.5, yPct: 0.54, isResult: false },
        { name: 'fashion5_food1_result_8.jpg', xPct: 0.72, yPct: 0.36, isResult: true },
        { name: 'food1', xPct: 0.23, yPct: 0.38, isResult: false },
        { name: 'sea4', xPct: -0.01, yPct: 0.41, isResult: false },
        { name: 'arch1', xPct: 0.22, yPct: 0.7, isResult: false },
        { name: 'arch1_food1_result_2.jpg', xPct: -0.03, yPct: 0.87, isResult: true },
        { name: 'food1_sea4_result_3.jpg', xPct: -0.23, yPct: 0.1, isResult: true }
    ];

    // Initialize the canvas
    async function initCanvas() {
        canvas = document.getElementById('demo-canvas');
        if (!canvas) return;

        ctx = canvas.getContext('2d');

        // Set canvas size based on container
        const container = canvas.parentElement;
        const rect = container.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;

        // Use percentage-based height relative to container width (aspect ratio)
        const canvasHeight = Math.min(rect.width * 0.75, 500); // 75% of width, max 500px

        canvas.width = rect.width * dpr;
        canvas.height = canvasHeight * dpr;
        canvas.style.width = rect.width + 'px';
        canvas.style.height = canvasHeight + 'px';

        ctx.scale(dpr, dpr);

        // Calculate image sizes as percentage of canvas
        const canvasDisplayWidth = rect.width;
        const canvasDisplayHeight = canvasHeight;
        const imgSize = Math.min(canvasDisplayWidth, canvasDisplayHeight) * 0.28;

        // Load initial images
        const initialImages = ['food1', 'nature2', 'fashion5', 'food1_nature2_result_2.jpg', 'fashion5_food1_result_8.jpg'];

        try {
            await Promise.all(initialImages.map(name => loadImage(name)));

            // Position images using percentage-based layout
            INITIAL_LAYOUT.forEach(item => {
                const x = item.xPct * canvasDisplayWidth;
                const y = item.yPct * canvasDisplayHeight;
                addCanvasImage(item.name, x, y, imgSize, imgSize, item.isResult);
            });

            // Store initial state for reset (as percentages for proper scaling)
            initialCanvasState = canvasImages.map(img => ({
                ...img,
                xPct: img.x / canvasDisplayWidth,
                yPct: img.y / canvasDisplayHeight
            }));

            render();
            updateSelectorVisibility();
            document.getElementById('image-selector').style.opacity = '1';
        } catch (err) {
            console.error('Failed to load images:', err);
        }

        // Add event listeners
        canvas.addEventListener('mousedown', onMouseDown);
        canvas.addEventListener('mousemove', onMouseMove);
        canvas.addEventListener('mouseup', onMouseUp);
        canvas.addEventListener('mouseleave', onMouseUp);
        canvas.addEventListener('click', onCanvasClick);

        canvas.addEventListener('touchstart', onMouseDown, { passive: false });
        canvas.addEventListener('touchmove', onMouseMove, { passive: false });
        canvas.addEventListener('touchend', onMouseUp);
        canvas.addEventListener('touchend', onCanvasClick);

        // Setup image selector
        const selectorImages = document.querySelectorAll('.selector-image');
        selectorImages.forEach(img => {
            img.addEventListener('click', onSelectorImageClick);
        });

        // Setup zoom controls
        const zoomInBtn = document.getElementById('zoom-in-btn');
        const zoomOutBtn = document.getElementById('zoom-out-btn');
        const resetBtn = document.getElementById('reset-btn');

        if (zoomInBtn) zoomInBtn.addEventListener('click', zoomIn);
        if (zoomOutBtn) zoomOutBtn.addEventListener('click', zoomOut);
        if (resetBtn) resetBtn.addEventListener('click', resetView);

        // Debug helper — call dumpCanvasState() in browser console to get current layout
        window.dumpCanvasState = function() {
            const w = parseFloat(canvas.style.width);
            const h = parseFloat(canvas.style.height);
            const layout = canvasImages.map(img => ({
                name: img.name,
                xPct: Math.round(img.x / w * 100) / 100,
                yPct: Math.round(img.y / h * 100) / 100,
                isResult: img.isResult
            }));
            console.log('INITIAL_LAYOUT:', JSON.stringify(layout, null, 4));
            console.log('zoomLevel:', zoomLevel);
        };

        // Handle resize
        window.addEventListener('resize', debounce(() => {
            const newRect = container.getBoundingClientRect();
            const newDpr = window.devicePixelRatio || 1;
            const newCanvasHeight = Math.min(newRect.width * 0.75, 500);

            // Store current positions as percentages before resize
            const oldWidth = parseFloat(canvas.style.width);
            const oldHeight = parseFloat(canvas.style.height);

            canvas.width = newRect.width * newDpr;
            canvas.height = newCanvasHeight * newDpr;
            canvas.style.width = newRect.width + 'px';
            canvas.style.height = newCanvasHeight + 'px';
            ctx.scale(newDpr, newDpr);

            // Scale image positions and sizes proportionally
            const scaleX = newRect.width / oldWidth;
            const scaleY = newCanvasHeight / oldHeight;
            const newImgSize = Math.min(newRect.width, newCanvasHeight) * 0.28;

            canvasImages.forEach(img => {
                const dims = getDisplayDimensions(img.img, newImgSize, newImgSize);
                img.x = img.x * scaleX;
                img.y = img.y * scaleY;
                img.width = dims.width;
                img.height = dims.height;
            });

            // Update initial state for reset
            initialCanvasState = initialCanvasState.map(img => {
                const dims = getDisplayDimensions(img.img, newImgSize, newImgSize);
                return {
                    ...img,
                    x: img.xPct * newRect.width,
                    y: img.yPct * newCanvasHeight,
                    width: dims.width,
                    height: dims.height
                };
            });

            render();
        }, 250));
    }

    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    // Log current image positions (for setting up initial layout)
    function logImagePositions() {
        console.log('Current image positions:');
        canvasImages.forEach(img => {
            console.log(`  { name: '${img.name}', x: ${Math.round(img.x)}, y: ${Math.round(img.y)} }`);
        });
    }

    // Expose to window for console access
    window.logImagePositions = logImagePositions;

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initCanvas);
    } else {
        initCanvas();
    }
})();
