// Annotation Reviewer JavaScript

class AnnotationReviewer {
    constructor() {
        this.currentTile = null;
        this.statistics = null;
        this.isLoading = false;
        this.classes = [];
        this.enabledClasses = new Set();
        this.selectedMaskId = null;
        this.selectedMaskForReassign = null;

        // Editing state
        this.editingEnabled = false;
        this.editMode = 'draw'; // 'draw', 'erase', or 'polygon'
        this.brushSize = 10;
        this.isDrawing = false;
        this.lastPoint = null;
        this.editedMaskData = null;
        this.originalMaskData = null;

        // Polygon state
        this.polygonPoints = [];
        this.isDrawingPolygon = false;
        this.lastClickTime = 0;
        this.lastClickPoint = null;

        // Initialize UI elements
        this.initElements();
        this.attachEventListeners();
        this.loadInitialData();
    }

    initElements() {
        // Images
        this.tileImage = document.getElementById('tile-image');
        this.maskOverlay = document.getElementById('mask-overlay');
        this.loadingSpinner = document.getElementById('loading-spinner');

        // Controls
        this.toggleMask = document.getElementById('toggle-mask');
        this.maskOpacity = document.getElementById('mask-opacity');
        this.opacityValue = document.getElementById('opacity-value');

        // Buttons
        this.btnAccept = document.getElementById('btn-accept');
        this.btnReject = document.getElementById('btn-reject');
        this.btnSkip = document.getElementById('btn-skip');
        this.btnPrev = document.getElementById('btn-prev');
        this.btnNext = document.getElementById('btn-next');
        this.btnSaveNotes = document.getElementById('btn-save-notes');
        this.btnExport = document.getElementById('btn-export');

        // Info displays
        this.tileTitle = document.getElementById('tile-title');
        this.tileCounter = document.getElementById('tile-counter');
        this.notesText = document.getElementById('notes-text');

        // Statistics
        this.statTotal = document.getElementById('stat-total');
        this.statAccepted = document.getElementById('stat-accepted');
        this.statRejected = document.getElementById('stat-rejected');
        this.statProgress = document.getElementById('stat-progress');

        // Classes
        this.classesSection = document.getElementById('classes-section');
        this.classesList = document.getElementById('classes-list');
        this.classCount = document.getElementById('class-count');
        this.btnSelectAll = document.getElementById('btn-select-all');
        this.btnDeselectAll = document.getElementById('btn-deselect-all');

        // Reassign section
        this.reassignSection = document.getElementById('reassign-section');
        this.reassignOptions = document.getElementById('reassign-options');
        this.btnApplyReassign = document.getElementById('btn-apply-reassign');

        // Editing
        this.editCanvas = document.getElementById('edit-canvas');
        this.editCtx = this.editCanvas.getContext('2d');
        this.btnToggleEdit = document.getElementById('btn-toggle-edit');
        this.editTools = document.getElementById('edit-tools');
        this.btnDrawMode = document.getElementById('btn-draw-mode');
        this.btnEraseMode = document.getElementById('btn-erase-mode');
        this.btnPolygonMode = document.getElementById('btn-polygon-mode');
        this.brushSizeInput = document.getElementById('brush-size');
        this.brushSizeValue = document.getElementById('brush-size-value');
        this.editClassSelect = document.getElementById('edit-class-select');
        this.btnCancelEdits = document.getElementById('btn-cancel-edits');

        // Polygon tools
        this.polygonTools = document.getElementById('polygon-tools');
        this.btnClearPolygon = document.getElementById('btn-clear-polygon');
    }

    attachEventListeners() {
        // Action buttons
        this.btnAccept.addEventListener('click', () => this.acceptTile());
        this.btnReject.addEventListener('click', () => this.rejectTile());
        this.btnSkip.addEventListener('click', () => this.skipTile());
        this.btnPrev.addEventListener('click', () => this.navigate('previous'));
        this.btnNext.addEventListener('click', () => this.navigate('next'));
        this.btnSaveNotes.addEventListener('click', () => this.saveNotes());
        this.btnExport.addEventListener('click', () => this.exportResults());

        // Mask controls
        this.toggleMask.addEventListener('change', () => this.updateMaskVisibility());
        this.maskOpacity.addEventListener('input', () => this.updateMaskOpacity());

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyPress(e));

        // Image load events
        this.tileImage.addEventListener('load', () => this.onImageLoad());
        this.tileImage.addEventListener('error', (e) => this.onImageError('tile', e));
        this.maskOverlay.addEventListener('load', () => this.onMaskLoad());
        this.maskOverlay.addEventListener('error', (e) => this.onImageError('mask', e));

        // Class controls
        this.btnSelectAll.addEventListener('click', () => this.selectAllClasses());
        this.btnDeselectAll.addEventListener('click', () => this.deselectAllClasses());

        // Reassign controls
        this.btnApplyReassign.addEventListener('click', () => this.applyReassignment());

        // Editing controls
        this.btnToggleEdit.addEventListener('click', () => this.toggleEditing());
        this.btnDrawMode.addEventListener('click', () => this.setEditMode('draw'));
        this.btnEraseMode.addEventListener('click', () => this.setEditMode('erase'));
        this.btnPolygonMode.addEventListener('click', () => this.setEditMode('polygon'));
        this.brushSizeInput.addEventListener('input', () => this.updateBrushSize());
        this.btnCancelEdits.addEventListener('click', () => this.cancelEdits());

        // Polygon controls
        this.btnClearPolygon.addEventListener('click', () => this.clearPolygon());

        // Canvas drawing events
        this.editCanvas.addEventListener('mousedown', (e) => this.handleCanvasMouseDown(e));
        this.editCanvas.addEventListener('mousemove', (e) => this.handleCanvasMouseMove(e));
        this.editCanvas.addEventListener('mouseup', () => this.stopDrawing());
        this.editCanvas.addEventListener('mouseleave', () => this.stopDrawing());

        // Touch events for mobile
        this.editCanvas.addEventListener('touchstart', (e) => this.handleCanvasMouseDown(e));
        this.editCanvas.addEventListener('touchmove', (e) => this.handleCanvasMouseMove(e));
        this.editCanvas.addEventListener('touchend', () => this.stopDrawing());
    }

    handleKeyPress(e) {
        // Don't trigger shortcuts when typing in textarea
        if (e.target.tagName === 'TEXTAREA') return;

        // Handle polygon mode keys
        if (this.editMode === 'polygon' && this.editingEnabled) {
            if (e.key === 'Enter') {
                e.preventDefault();
                this.finishPolygon();
                return;
            }
            if (e.key === 'Escape') {
                e.preventDefault();
                this.clearPolygon();
                return;
            }
        }

        switch(e.key.toLowerCase()) {
            case 'a':
                e.preventDefault();
                this.acceptTile();
                break;
            case 'r':
                e.preventDefault();
                this.rejectTile();
                break;
            case 's':
                e.preventDefault();
                this.skipTile();
                break;
            case 'm':
                e.preventDefault();
                this.toggleMask.checked = !this.toggleMask.checked;
                this.updateMaskVisibility();
                break;
            case 'arrowleft':
                e.preventDefault();
                this.navigate('previous');
                break;
            case 'arrowright':
                e.preventDefault();
                this.navigate('next');
                break;
        }
    }

    async loadInitialData() {
        await this.updateStatus();
        await this.loadCurrentTile();
    }

    async updateStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();

            if (data.success) {
                this.statistics = data.statistics;
                this.updateStatisticsDisplay();
            }
        } catch (error) {
            console.error('Error updating status:', error);
            this.showError('Failed to load status');
        }
    }

    updateStatisticsDisplay() {
        if (!this.statistics) return;

        const counts = this.statistics.counts;
        this.statTotal.textContent = counts.total;
        this.statAccepted.textContent = counts.accepted;
        this.statRejected.textContent = counts.rejected;
        this.statProgress.textContent = this.statistics.progress_percent.toFixed(1) + '%';
    }

    async loadCurrentTile() {
        if (this.isLoading) return;

        this.isLoading = true;
        this.showLoading();

        try {
            // Get tile info
            const response = await fetch('/api/tile/current');
            const data = await response.json();

            if (!data.success) {
                this.showError(data.error || 'No tile available');
                this.isLoading = false;
                return;
            }

            this.currentTile = data;

            // Update UI
            this.tileTitle.textContent = `Tile: ${data.tile_id}`;
            this.tileCounter.textContent = `${data.index + 1} / ${data.total}`;
            this.notesText.value = data.notes || '';

            // Load images
            const timestamp = new Date().getTime(); // Cache busting
            console.log('Loading tile image:', `/api/image/tile?t=${timestamp}`);
            this.tileImage.src = `/api/image/tile?t=${timestamp}`;

            // Load classes
            await this.loadClasses();

            // Load mask with classes
            this.updateMaskDisplay();

            // Update status after loading tile
            await this.updateStatus();

            // Safety timeout: hide loading spinner after 5 seconds even if images haven't loaded
            setTimeout(() => {
                if (this.isLoading) {
                    console.warn('Image loading timeout - forcing hide loading spinner');
                    this.hideLoading();
                }
            }, 5000);

        } catch (error) {
            console.error('Error loading tile:', error);
            this.showError('Failed to load tile');
        } finally {
            this.isLoading = false;
        }
    }

    async loadClasses() {
        try {
            const response = await fetch('/api/mask/classes');
            const data = await response.json();

            if (data.success) {
                this.classes = data.classes;

                // Initialize all classes as enabled
                this.enabledClasses.clear();
                this.classes.forEach(cls => this.enabledClasses.add(cls.id));

                // Update UI
                this.displayClasses();
            }
        } catch (error) {
            console.error('Error loading classes:', error);
            this.classes = [];
            this.displayClasses();
        }
    }

    displayClasses() {
        if (this.classes.length === 0) {
            this.classesSection.style.display = 'none';
            this.reassignSection.style.display = 'none';
            return;
        }

        this.classesSection.style.display = 'block';
        this.classCount.textContent = `(${this.classes.length})`;

        // Clear existing classes
        this.classesList.innerHTML = '';

        // Add each class
        this.classes.forEach(cls => {
            const classItem = document.createElement('div');
            classItem.className = 'class-item';
            classItem.dataset.classId = cls.id;

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.id = `class-${cls.id}`;
            checkbox.checked = this.enabledClasses.has(cls.id);
            checkbox.addEventListener('change', () => this.toggleClass(cls.id));

            const colorBox = document.createElement('span');
            colorBox.className = 'class-color';
            colorBox.style.backgroundColor = `rgba(${cls.color[0]}, ${cls.color[1]}, ${cls.color[2]}, ${cls.color[3] / 255})`;

            const label = document.createElement('label');
            label.htmlFor = `class-${cls.id}`;
            label.textContent = `${cls.label} (${cls.area} px)`;

            const selectBtn = document.createElement('button');
            selectBtn.className = 'btn-select-mask';
            selectBtn.textContent = '⊙';
            selectBtn.title = 'Select for reassignment';
            selectBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.selectMaskForReassignment(cls.id, cls.label);
            });

            const deleteBtn = document.createElement('button');
            deleteBtn.className = 'btn-delete-mask';
            deleteBtn.textContent = '✗';
            deleteBtn.title = 'Delete this mask';
            deleteBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.deleteMask(cls.id);
            });

            classItem.appendChild(checkbox);
            classItem.appendChild(colorBox);
            classItem.appendChild(label);
            classItem.appendChild(selectBtn);
            classItem.appendChild(deleteBtn);

            this.classesList.appendChild(classItem);
        });

        // Update reassign radio buttons if needed
        if (this.selectedMaskForReassign !== null) {
            this.showReassignOptions();
        }
    }

    toggleClass(classId) {
        const checkbox = document.getElementById(`class-${classId}`);
        if (checkbox.checked) {
            this.enabledClasses.add(classId);
        } else {
            this.enabledClasses.delete(classId);
        }
        this.updateMaskDisplay();
    }

    selectAllClasses() {
        this.classes.forEach(cls => {
            this.enabledClasses.add(cls.id);
            const checkbox = document.getElementById(`class-${cls.id}`);
            if (checkbox) checkbox.checked = true;
        });
        this.updateMaskDisplay();
    }

    deselectAllClasses() {
        this.enabledClasses.clear();
        this.classes.forEach(cls => {
            const checkbox = document.getElementById(`class-${cls.id}`);
            if (checkbox) checkbox.checked = false;
        });
        this.updateMaskDisplay();
    }

    updateMaskDisplay() {
        if (this.classes.length === 0) {
            // Fall back to old single-mask display
            const timestamp = new Date().getTime();
            const url = `/api/image/mask?t=${timestamp}`;
            console.log('Loading mask image:', url);
            this.maskOverlay.src = url;
        } else {
            // Use multi-class display
            const enabledIds = Array.from(this.enabledClasses).join(',');
            const timestamp = new Date().getTime();
            const url = `/api/image/mask_multiclass?enabled=${enabledIds}&t=${timestamp}`;
            console.log('Loading mask image:', url);
            this.maskOverlay.src = url;
        }
    }

    async deleteMask(maskId) {
        if (!confirm(`Delete this mask instance?`)) {
            return;
        }

        try {
            const response = await fetch('/api/mask/delete', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ mask_id: maskId })
            });

            const data = await response.json();

            if (data.success) {
                // Reload classes and mask
                await this.loadClasses();
                this.updateMaskDisplay();
                this.showMessage('Mask deleted');
            } else {
                alert(`Failed to delete mask: ${data.error}`);
            }
        } catch (error) {
            console.error('Error deleting mask:', error);
            alert('Failed to delete mask');
        }
    }

    selectMaskForReassignment(maskId, currentLabel) {
        this.selectedMaskForReassign = { id: maskId, label: currentLabel };
        this.showReassignOptions();
    }

    showReassignOptions() {
        if (!this.selectedMaskForReassign) return;

        // Get unique labels from prompts
        const uniqueLabels = new Set();
        this.classes.forEach(cls => uniqueLabels.add(cls.label));

        // Show reassign section
        this.reassignSection.style.display = 'block';

        // Clear and populate radio buttons
        this.reassignOptions.innerHTML = '';

        const currentLabel = this.selectedMaskForReassign.label;

        // Add radio buttons for each unique label
        uniqueLabels.forEach(label => {
            const radioDiv = document.createElement('div');
            radioDiv.className = 'reassign-option';

            const radio = document.createElement('input');
            radio.type = 'radio';
            radio.name = 'reassign-category';
            radio.id = `reassign-${label}`;
            radio.value = label;
            radio.checked = (label === currentLabel);

            const radioLabel = document.createElement('label');
            radioLabel.htmlFor = `reassign-${label}`;
            radioLabel.textContent = label;
            if (label === currentLabel) {
                radioLabel.style.fontWeight = 'bold';
                radioLabel.textContent += ' (current)';
            }

            radioDiv.appendChild(radio);
            radioDiv.appendChild(radioLabel);
            this.reassignOptions.appendChild(radioDiv);
        });

        // Add "new category" option
        const newCategoryDiv = document.createElement('div');
        newCategoryDiv.className = 'reassign-option';

        const newRadio = document.createElement('input');
        newRadio.type = 'radio';
        newRadio.name = 'reassign-category';
        newRadio.id = 'reassign-new';
        newRadio.value = '__new__';

        const newLabel = document.createElement('label');
        newLabel.htmlFor = 'reassign-new';
        newLabel.textContent = 'New category...';

        newCategoryDiv.appendChild(newRadio);
        newCategoryDiv.appendChild(newLabel);
        this.reassignOptions.appendChild(newCategoryDiv);

        this.btnApplyReassign.style.display = 'block';
    }

    async applyReassignment() {
        if (!this.selectedMaskForReassign) return;

        const selectedRadio = document.querySelector('input[name="reassign-category"]:checked');
        if (!selectedRadio) {
            alert('Please select a category');
            return;
        }

        let newLabel = selectedRadio.value;

        if (newLabel === '__new__') {
            newLabel = prompt('Enter new category name:');
            if (!newLabel) return;
        }

        if (newLabel === this.selectedMaskForReassign.label) {
            this.showMessage('No change needed');
            return;
        }

        try {
            const response = await fetch('/api/mask/reassign', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    mask_id: this.selectedMaskForReassign.id,
                    new_label: newLabel
                })
            });

            const data = await response.json();

            if (data.success) {
                // Clear selection and hide reassign section
                this.selectedMaskForReassign = null;
                this.reassignSection.style.display = 'none';

                // Reload classes and mask
                await this.loadClasses();
                this.updateMaskDisplay();
                this.showMessage(`Mask reassigned to ${newLabel}`);
            } else {
                alert(`Failed to reassign mask: ${data.error}`);
            }
        } catch (error) {
            console.error('Error reassigning mask:', error);
            alert('Failed to reassign mask');
        }
    }

    async acceptTile() {
        await this.submitAction('accept', 'Accepting...');
    }

    async rejectTile() {
        await this.submitAction('reject', 'Rejecting...');
    }

    async skipTile() {
        await this.submitAction('skip', 'Skipping...');
    }

    async submitAction(action, loadingMessage) {
        if (this.isLoading) return;

        this.showLoading(loadingMessage);

        try {
            const notes = this.notesText.value;
            const response = await fetch(`/api/tile/${action}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ notes })
            });

            const data = await response.json();

            if (data.success) {
                await this.loadCurrentTile();
            } else {
                this.showError(data.error || `Failed to ${action} tile`);
            }
        } catch (error) {
            console.error(`Error ${action}ing tile:`, error);
            this.showError(`Failed to ${action} tile`);
        }
    }

    async navigate(direction) {
        if (this.isLoading) return;

        this.showLoading('Loading...');

        try {
            const response = await fetch('/api/tile/navigate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ direction })
            });

            const data = await response.json();

            if (data.success) {
                await this.loadCurrentTile();
            } else {
                this.showError(data.error || 'Failed to navigate');
            }
        } catch (error) {
            console.error('Error navigating:', error);
            this.showError('Failed to navigate');
        }
    }

    async saveNotes() {
        try {
            const notes = this.notesText.value;
            const response = await fetch('/api/tile/update_notes', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ notes })
            });

            const data = await response.json();

            if (data.success) {
                this.showMessage('Notes saved');
            } else {
                this.showError(data.error || 'Failed to save notes');
            }
        } catch (error) {
            console.error('Error saving notes:', error);
            this.showError('Failed to save notes');
        }
    }

    async exportResults() {
        const outputPath = prompt('Enter output directory path:', './reviewed_masks');
        if (!outputPath) return;

        try {
            const response = await fetch('/api/export', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ output_path: outputPath })
            });

            const data = await response.json();

            if (data.success) {
                alert(`Results exported successfully to ${outputPath}`);
            } else {
                alert(`Export failed: ${data.error}`);
            }
        } catch (error) {
            console.error('Error exporting:', error);
            alert('Failed to export results');
        }
    }

    updateMaskVisibility() {
        if (this.toggleMask.checked) {
            this.maskOverlay.classList.remove('hidden');
        } else {
            this.maskOverlay.classList.add('hidden');
        }
    }

    updateMaskOpacity() {
        const opacity = this.maskOpacity.value / 100;
        this.maskOverlay.style.opacity = opacity;
        this.opacityValue.textContent = this.maskOpacity.value + '%';
    }

    showLoading(message = 'Loading...') {
        this.loadingSpinner.textContent = message;
        this.loadingSpinner.classList.remove('hidden');
        this.disableButtons();
    }

    hideLoading() {
        this.loadingSpinner.classList.add('hidden');
        this.enableButtons();
    }

    onImageLoad() {
        console.log('Tile image loaded');
        this.checkImagesLoaded();
    }

    onMaskLoad() {
        console.log('Mask image loaded');
        this.checkImagesLoaded();
    }

    onImageError(imageType, error) {
        console.error(`Error loading ${imageType} image:`, error);
        // Still check if we can proceed
        this.checkImagesLoaded();
    }

    checkImagesLoaded() {
        // Hide loading if both images are in a final state (loaded or errored)
        const tileReady = this.tileImage.complete;
        const maskReady = this.maskOverlay.complete;

        console.log(`Images ready: tile=${tileReady}, mask=${maskReady}`);

        if (tileReady && maskReady) {
            this.hideLoading();
        }
    }

    disableButtons() {
        this.btnAccept.disabled = true;
        this.btnReject.disabled = true;
        this.btnSkip.disabled = true;
        this.btnPrev.disabled = true;
        this.btnNext.disabled = true;
    }

    enableButtons() {
        this.btnAccept.disabled = false;
        this.btnReject.disabled = false;
        this.btnSkip.disabled = false;
        this.btnPrev.disabled = false;
        this.btnNext.disabled = false;
    }

    showError(message) {
        console.error(message);
        this.tileTitle.textContent = `Error: ${message}`;
        this.hideLoading();
    }

    showMessage(message) {
        // Simple message display - could be enhanced with a toast notification
        console.log(message);
    }

    // ========== EDITING METHODS ==========

    toggleEditing() {
        this.editingEnabled = !this.editingEnabled;

        if (this.editingEnabled) {
            this.btnToggleEdit.textContent = 'Disable Editing';
            this.btnToggleEdit.classList.add('active');
            this.editTools.style.display = 'block';
            this.editCanvas.classList.add('active');
            this.initializeCanvas();
            this.loadMaskForEditing();
        } else {
            this.btnToggleEdit.textContent = 'Enable Editing';
            this.btnToggleEdit.classList.remove('active');
            this.editTools.style.display = 'none';
            this.editCanvas.classList.remove('active');
            this.clearCanvas();
        }
    }

    initializeCanvas() {
        // Size canvas to match mask overlay
        const img = this.maskOverlay;
        this.editCanvas.width = img.naturalWidth || 512;
        this.editCanvas.height = img.naturalHeight || 512;

        // Update canvas style to match image display
        const rect = img.getBoundingClientRect();
        this.editCanvas.style.width = rect.width + 'px';
        this.editCanvas.style.height = rect.height + 'px';
    }

    async loadMaskForEditing() {
        try {
            const response = await fetch('/api/mask/data');
            const data = await response.json();

            if (data.success) {
                this.originalMaskData = data.mask_data;
                this.editedMaskData = JSON.parse(JSON.stringify(data.mask_data)); // Deep copy
                this.updateEditClassSelect();
                this.redrawCanvas();
            }
        } catch (error) {
            console.error('Error loading mask for editing:', error);
        }
    }

    updateEditClassSelect() {
        // Populate class dropdown
        this.editClassSelect.innerHTML = '<option value="">All classes (merged)</option>';

        this.classes.forEach(cls => {
            const option = document.createElement('option');
            option.value = cls.id;
            option.textContent = cls.label;
            this.editClassSelect.appendChild(option);
        });
    }

    setEditMode(mode) {
        this.editMode = mode;
        this.btnDrawMode.classList.toggle('active', mode === 'draw');
        this.btnEraseMode.classList.toggle('active', mode === 'erase');
        this.btnPolygonMode.classList.toggle('active', mode === 'polygon');

        // Show/hide brush size in polygon mode
        const brushSizeGroup = this.brushSizeInput.closest('.tool-group');
        if (brushSizeGroup) {
            brushSizeGroup.style.display = (mode === 'polygon') ? 'none' : 'block';
        }

        // Show/hide polygon tools
        if (mode === 'polygon') {
            this.polygonTools.style.display = 'block';
            this.clearPolygon();
        } else {
            this.polygonTools.style.display = 'none';
            this.clearPolygon();
        }
    }

    updateBrushSize() {
        this.brushSize = parseInt(this.brushSizeInput.value);
        this.brushSizeValue.textContent = this.brushSize;
    }

    handleCanvasMouseDown(e) {
        if (!this.editingEnabled) return;

        e.preventDefault();

        // Handle polygon mode differently
        if (this.editMode === 'polygon') {
            this.addPolygonPoint(e);
            return;
        }

        // Regular draw/erase mode
        this.startDrawing(e);
    }

    handleCanvasMouseMove(e) {
        if (!this.editingEnabled) return;

        // In polygon mode, just show cursor position
        if (this.editMode === 'polygon') {
            if (this.polygonPoints.length > 0) {
                this.drawPolygonPreview(e);
            }
            return;
        }

        // Regular draw/erase mode
        this.draw(e);
    }

    startDrawing(e) {
        if (!this.editingEnabled) return;

        e.preventDefault();

        this.isDrawing = true;
        const point = this.getCanvasPoint(e);
        this.lastPoint = point;
        this.drawAtPoint(point);
    }

    draw(e) {
        if (!this.isDrawing) return;

        e.preventDefault();
        const point = this.getCanvasPoint(e);

        if (this.lastPoint) {
            this.drawLine(this.lastPoint, point);
        }

        this.lastPoint = point;
    }

    stopDrawing() {
        if (this.editMode === 'polygon') {
            // Don't clear polygon on mouse up
            return;
        }
        this.isDrawing = false;
        this.lastPoint = null;
    }

    getCanvasPoint(e) {
        const rect = this.editCanvas.getBoundingClientRect();
        const scaleX = this.editCanvas.width / rect.width;
        const scaleY = this.editCanvas.height / rect.height;

        let clientX, clientY;

        if (e.touches && e.touches.length > 0) {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
        } else {
            clientX = e.clientX;
            clientY = e.clientY;
        }

        return {
            x: (clientX - rect.left) * scaleX,
            y: (clientY - rect.top) * scaleY
        };
    }

    drawAtPoint(point) {
        const ctx = this.editCtx;
        const radius = this.brushSize / 2;

        // Set drawing style
        if (this.editMode === 'draw') {
            // Get color for current class
            const classId = this.editClassSelect.value;
            if (classId === '') {
                ctx.fillStyle = 'rgba(0, 255, 255, 0.7)'; // Default cyan
            } else {
                const cls = this.classes.find(c => c.id == classId);
                if (cls) {
                    ctx.fillStyle = `rgba(${cls.color[0]}, ${cls.color[1]}, ${cls.color[2]}, 0.7)`;
                }
            }
        } else if (this.editMode === 'erase') {
            // Eraser - use destination-out to make transparent
            ctx.globalCompositeOperation = 'destination-out';
            ctx.fillStyle = 'rgba(0, 0, 0, 1)';
        }

        // Draw circle
        ctx.beginPath();
        ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
        ctx.fill();

        // Reset composite operation
        ctx.globalCompositeOperation = 'source-over';

        // Update mask data
        this.updateMaskData(point);
    }

    drawLine(p1, p2) {
        // Bresenham's line algorithm
        const dx = Math.abs(p2.x - p1.x);
        const dy = Math.abs(p2.y - p1.y);
        const sx = p1.x < p2.x ? 1 : -1;
        const sy = p1.y < p2.y ? 1 : -1;
        let err = dx - dy;

        let x = p1.x;
        let y = p1.y;

        while (true) {
            this.drawAtPoint({ x, y });

            if (Math.abs(x - p2.x) < 1 && Math.abs(y - p2.y) < 1) break;

            const e2 = 2 * err;
            if (e2 > -dy) {
                err -= dy;
                x += sx;
            }
            if (e2 < dx) {
                err += dx;
                y += sy;
            }
        }
    }

    updateMaskData(point) {
        if (!this.editedMaskData || !this.editedMaskData.masks) return;

        const classId = this.editClassSelect.value;
        const radius = this.brushSize / 2;
        const x = Math.round(point.x);
        const y = Math.round(point.y);
        const isDraw = (this.editMode === 'draw');

        // Update pixels in circular area
        for (let dy = -radius; dy <= radius; dy++) {
            for (let dx = -radius; dx <= radius; dx++) {
                if (dx * dx + dy * dy <= radius * radius) {
                    const px = x + dx;
                    const py = y + dy;

                    if (px >= 0 && px < this.editCanvas.width && py >= 0 && py < this.editCanvas.height) {
                        if (classId === '') {
                            // Update all masks
                            this.editedMaskData.masks.forEach(mask => {
                                const idx = py * this.editCanvas.width + px;
                                mask[idx] = isDraw ? 1 : 0;
                            });
                        } else {
                            // Update specific class
                            const maskIdx = parseInt(classId);
                            if (this.editedMaskData.masks[maskIdx]) {
                                const idx = py * this.editCanvas.width + px;
                                this.editedMaskData.masks[maskIdx][idx] = isDraw ? 1 : 0;
                            }
                        }
                    }
                }
            }
        }
    }

    redrawCanvas() {
        this.clearCanvas();

        if (!this.editedMaskData || !this.editedMaskData.masks) return;

        const ctx = this.editCtx;
        const imageData = ctx.createImageData(this.editCanvas.width, this.editCanvas.height);

        // Draw each mask with its color
        this.editedMaskData.masks.forEach((mask, i) => {
            const cls = this.classes[i];
            if (!cls) return;

            const color = cls.color;

            for (let y = 0; y < this.editCanvas.height; y++) {
                for (let x = 0; x < this.editCanvas.width; x++) {
                    const idx = y * this.editCanvas.width + x;
                    if (mask[idx] > 0) {
                        const pixelIdx = idx * 4;
                        imageData.data[pixelIdx] = color[0];
                        imageData.data[pixelIdx + 1] = color[1];
                        imageData.data[pixelIdx + 2] = color[2];
                        imageData.data[pixelIdx + 3] = color[3];
                    }
                }
            }
        });

        ctx.putImageData(imageData, 0, 0);
    }

    clearCanvas() {
        this.editCtx.clearRect(0, 0, this.editCanvas.width, this.editCanvas.height);
    }

    async saveEdits() {
        if (!this.editedMaskData) {
            alert('No edits to save');
            return;
        }

        try {
            const response = await fetch('/api/mask/save_edits', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    mask_data: this.editedMaskData
                })
            });

            const data = await response.json();

            if (data.success) {
                alert('Edits saved successfully!');
                this.originalMaskData = JSON.parse(JSON.stringify(this.editedMaskData));
                // Reload classes and mask display
                await this.loadClasses();
                this.updateMaskDisplay();
            } else {
                alert(`Failed to save edits: ${data.error}`);
            }
        } catch (error) {
            console.error('Error saving edits:', error);
            alert('Failed to save edits');
        }
    }

    cancelEdits() {
        if (confirm('Discard all edits?')) {
            this.editedMaskData = JSON.parse(JSON.stringify(this.originalMaskData));
            this.redrawCanvas();
        }
    }

    // ========== POLYGON DRAWING METHODS ==========

    addPolygonPoint(e) {
        const point = this.getCanvasPoint(e);
        const currentTime = Date.now();

        // Check for double-click (within 300ms)
        const isDoubleClick = this.lastClickPoint &&
            (currentTime - this.lastClickTime) < 300 &&
            Math.sqrt(Math.pow(point.x - this.lastClickPoint.x, 2) +
                     Math.pow(point.y - this.lastClickPoint.y, 2)) < 5;

        if (isDoubleClick && this.polygonPoints.length >= 3) {
            // Double-click detected, finish polygon
            this.finishPolygon();
            return;
        }

        // Check if clicking near first point to close polygon
        if (this.polygonPoints.length >= 3) {
            const firstPoint = this.polygonPoints[0];
            const distToFirst = Math.sqrt(
                Math.pow(point.x - firstPoint.x, 2) + Math.pow(point.y - firstPoint.y, 2)
            );
            if (distToFirst < 15) {  // Increased threshold for easier clicking
                this.finishPolygon();
                return;
            }
        }

        // Add point to polygon
        this.polygonPoints.push(point);
        this.lastClickTime = currentTime;
        this.lastClickPoint = point;

        // Draw polygon on canvas
        this.drawPolygonOnCanvas();
    }

    drawPolygonInProgress() {
        if (this.polygonPoints.length === 0) return;

        const ctx = this.editCtx;

        // Get color for current class
        const classId = this.editClassSelect.value;
        let color = [0, 255, 255]; // Default cyan

        if (classId !== '') {
            const cls = this.classes.find(c => c.id == classId);
            if (cls) {
                color = cls.color;
            }
        }

        // Draw polygon outline
        ctx.strokeStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, 1)`;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(this.polygonPoints[0].x, this.polygonPoints[0].y);
        for (let i = 1; i < this.polygonPoints.length; i++) {
            ctx.lineTo(this.polygonPoints[i].x, this.polygonPoints[i].y);
        }
        ctx.stroke();

        // Draw regular points
        ctx.fillStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, 1)`;
        this.polygonPoints.forEach((point, i) => {
            if (i !== 0) {  // Don't draw first point yet
                ctx.beginPath();
                ctx.arc(point.x, point.y, 4, 0, Math.PI * 2);
                ctx.fill();
            }
        });

        // Draw first point with special highlighting - always visible
        if (this.polygonPoints.length > 0) {
            const firstPoint = this.polygonPoints[0];

            // Draw larger circle for first point
            ctx.fillStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, 1)`;
            ctx.beginPath();
            ctx.arc(firstPoint.x, firstPoint.y, 6, 0, Math.PI * 2);
            ctx.fill();

            // Draw white outline
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.arc(firstPoint.x, firstPoint.y, 9, 0, Math.PI * 2);
            ctx.stroke();

            // If polygon can be closed, add pulsing effect
            if (this.polygonPoints.length >= 3) {
                ctx.strokeStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, 0.5)`;
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.arc(firstPoint.x, firstPoint.y, 12, 0, Math.PI * 2);
                ctx.stroke();
            }
        }
    }

    drawPolygonOnCanvas() {
        // Clear and redraw polygon
        this.clearCanvas();
        this.drawPolygonInProgress();
    }

    drawPolygonPreview(e) {
        if (this.polygonPoints.length === 0) return;

        // Redraw polygon with preview line
        this.clearCanvas();
        this.drawPolygonInProgress();

        const ctx = this.editCtx;
        const point = this.getCanvasPoint(e);

        // Get color for current class
        const classId = this.editClassSelect.value;
        let color = [0, 255, 255]; // Default cyan

        if (classId !== '') {
            const cls = this.classes.find(c => c.id == classId);
            if (cls) {
                color = cls.color;
            }
        }

        // Draw line from last point to cursor
        ctx.strokeStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, 0.5)`;
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        const lastPoint = this.polygonPoints[this.polygonPoints.length - 1];
        ctx.moveTo(lastPoint.x, lastPoint.y);
        ctx.lineTo(point.x, point.y);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    async finishPolygon() {
        if (this.polygonPoints.length < 3) {
            alert('Need at least 3 points to create a polygon');
            return;
        }

        // Make sure we have mask data loaded
        if (!this.editedMaskData || !this.editedMaskData.masks || this.editedMaskData.masks.length === 0) {
            alert('Please load mask data first by enabling editing mode.');
            return;
        }

        // Fill polygon in mask data
        this.fillPolygonInMask(this.polygonPoints);

        // Clear polygon points
        this.clearPolygon();

        // Redraw canvas with updated mask
        this.redrawCanvas();

        // Auto-save to server
        await this.saveEditsToServer();

        console.log('Polygon saved successfully');
    }

    async saveEditsToServer() {
        if (!this.editedMaskData) {
            return;
        }

        try {
            const response = await fetch('/api/mask/save_edits', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    mask_data: this.editedMaskData
                })
            });

            const data = await response.json();

            if (!data.success) {
                console.error('Failed to auto-save edits:', data.error);
            }
        } catch (error) {
            console.error('Error auto-saving edits:', error);
        }
    }

    clearPolygon() {
        this.polygonPoints = [];
        this.clearCanvas();
    }

    fillPolygonInMask(points) {
        if (!this.editedMaskData || !this.editedMaskData.masks) return;

        const classId = this.editClassSelect.value;
        // Polygon mode defaults to drawing (adding mask pixels)
        const isDraw = (this.editMode !== 'erase');

        // Create a path from points for point-in-polygon test
        // Using scanline algorithm for efficient polygon filling
        const minY = Math.floor(Math.min(...points.map(p => p.y)));
        const maxY = Math.ceil(Math.max(...points.map(p => p.y)));

        for (let y = minY; y <= maxY; y++) {
            // Find intersections with polygon edges at this y
            const intersections = [];
            for (let i = 0; i < points.length; i++) {
                const p1 = points[i];
                const p2 = points[(i + 1) % points.length];

                if ((p1.y <= y && p2.y > y) || (p2.y <= y && p1.y > y)) {
                    const x = p1.x + (y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y);
                    intersections.push(x);
                }
            }

            // Sort intersections
            intersections.sort((a, b) => a - b);

            // Fill between pairs of intersections
            for (let i = 0; i < intersections.length; i += 2) {
                if (i + 1 < intersections.length) {
                    const x1 = Math.floor(intersections[i]);
                    const x2 = Math.ceil(intersections[i + 1]);

                    for (let x = x1; x <= x2; x++) {
                        if (x >= 0 && x < this.editCanvas.width && y >= 0 && y < this.editCanvas.height) {
                            const idx = Math.floor(y) * this.editCanvas.width + x;

                            if (classId === '') {
                                // Update all masks
                                this.editedMaskData.masks.forEach(mask => {
                                    mask[idx] = isDraw ? 1 : 0;
                                });
                            } else {
                                // Update specific class
                                const maskIdx = parseInt(classId);
                                if (this.editedMaskData.masks[maskIdx]) {
                                    this.editedMaskData.masks[maskIdx][idx] = isDraw ? 1 : 0;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Initialize the reviewer when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new AnnotationReviewer();
});
