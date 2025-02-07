export class SamplingControls {
    constructor(container, onSamplingRequest) {
        this.container = container;
        this.onSamplingRequest = onSamplingRequest;
        this.isGenerating = false;
        this.lastUpdateTime = null;
        this.initialize();
    }
    initialize() {
        this.container.innerHTML = this.createTemplate();
        this.setupEventListeners();
        this.updateStatus('Idle');
    }
    createTemplate() {
        return `
            <div class="sampling-controls panel">
                <div class="panel-header">Sampling Controls</div>
                <div class="panel-content">
                    <div class="control-group">
                        <label class="control-label">Sampling Method</label>
                        <select class="sampling-method-select" id="sampling-method">
                            <option value="forward">Forward Sampling</option>
                            <option value="rejection">Rejection Sampling</option>
                            <option value="likelihood">Likelihood Sampling</option>
                            <option value="gibbs">Gibbs Sampling</option>
                        </select>
                    </div>

                    <div class="control-group">
                        <label class="control-label">Sample Size</label>
                        <input type="number" class="sample-size-input" id="sample-size" 
                               value="1000" min="1" max="100000">
                        <div class="preset-buttons">
                            <button class="preset-button" data-size="100">100</button>
                            <button class="preset-button" data-size="1000">1K</button>
                            <button class="preset-button" data-size="10000">10K</button>
                        </div>
                    </div>

                    <div class="control-group">
                        <div class="toggle-container">
                            <label class="toggle-switch">
                                <input type="checkbox" id="auto-update">
                                <span class="toggle-slider"></span>
                            </label>
                            <span class="control-label">Auto Update</span>
                        </div>
                        <div class="helper-text">Auto-updates limited to 1000 samples</div>
                    </div>

                    <div class="advanced-options">
                        <div class="advanced-header" id="advanced-toggle">
                            <span>Advanced Options</span>
                            <span class="toggle-icon">▼</span>
                        </div>
                        <div class="advanced-content" id="advanced-content">
                            <div class="control-group gibbs-only" style="display: none;">
                                <label class="control-label">Burn-in Period</label>
                                <input type="number" class="sample-size-input" id="burn-in" 
                                       value="100" min="0">
                            </div>
                            <div class="control-group gibbs-only" style="display: none;">
                                <label class="control-label">Thinning</label>
                                <input type="number" class="sample-size-input" id="thinning" 
                                       value="1" min="1">
                            </div>
                            <div class="control-group">
                                <label class="control-label">Random Seed (optional)</label>
                                <input type="number" class="sample-size-input" id="random-seed" 
                                       placeholder="Leave blank for random">
                            </div>
                            <div class="control-group">
                                <div class="toggle-container">
                                    <label class="toggle-switch">
                                        <input type="checkbox" id="cache-results" checked>
                                        <span class="toggle-slider"></span>
                                    </label>
                                    <span class="control-label">Cache Results</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <button class="generate-button" id="generate-button">
                        Generate Samples
                    </button>

                    <div class="status-area">
                        <div class="status-text" id="status-text">Status: Idle</div>
                        <div class="last-update" id="last-update"></div>
                    </div>

                    <div class="stats-section">
                        <div class="stats-grid">
                            <div class="stat-item">
                                <span>Total Samples:</span>
                                <span id="stat-total">0</span>
                            </div>
                            <div class="stat-item">
                                <span>Accepted:</span>
                                <span id="stat-accepted">0 (0%)</span>
                            </div>
                            <div class="stat-item">
                                <span>Rejected:</span>
                                <span id="stat-rejected">0 (0%)</span>
                            </div>
                            <div class="stat-item">
                                <span>Generation Time:</span>
                                <span id="stat-time">0s</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    setupEventListeners() {
        // Method selector
        const methodSelect = this.container.querySelector('#sampling-method');
        methodSelect.addEventListener('change', () => {
            this.toggleGibbsOptions(methodSelect.value === 'gibbs');
        });
        // Sample size presets
        this.container.querySelectorAll('.preset-button').forEach(button => {
            button.addEventListener('click', () => {
                const size = parseInt(button.dataset.size || '1000');
                this.container.querySelector('#sample-size').value = size.toString();
            });
        });
        // Advanced options toggle
        const advancedToggle = this.container.querySelector('#advanced-toggle');
        const advancedContent = this.container.querySelector('#advanced-content');
        advancedToggle.addEventListener('click', () => {
            advancedContent.classList.toggle('expanded');
            const toggleIcon = advancedToggle.querySelector('.toggle-icon');
            toggleIcon.textContent = advancedContent.classList.contains('expanded') ? '▼' : '▶';
        });
        // Auto-update toggle
        const autoUpdateToggle = this.container.querySelector('#auto-update');
        autoUpdateToggle.addEventListener('change', () => {
            if (autoUpdateToggle.checked) {
                const sampleSizeInput = this.container.querySelector('#sample-size');
                const sampleSize = parseInt(sampleSizeInput.value);
                if (sampleSize > 1000) {
                    sampleSizeInput.value = '1000';
                }
            }
        });
        // Sample size input validation
        const sampleSizeInput = this.container.querySelector('#sample-size');
        sampleSizeInput.addEventListener('change', () => {
            const autoUpdate = this.container.querySelector('#auto-update').checked;
            if (autoUpdate && parseInt(sampleSizeInput.value) > 1000) {
                sampleSizeInput.value = '1000';
            }
        });
        // Generate button
        const generateButton = this.container.querySelector('#generate-button');
        generateButton.addEventListener('click', () => this.generateSamples());
    }
    toggleGibbsOptions(show) {
        this.container.querySelectorAll('.gibbs-only').forEach(el => {
            el.style.display = show ? 'block' : 'none';
        });
    }
    getSettings() {
        return {
            method: this.container.querySelector('#sampling-method').value,
            sampleSize: parseInt(this.container.querySelector('#sample-size').value),
            autoUpdate: this.container.querySelector('#auto-update').checked,
            burnIn: parseInt(this.container.querySelector('#burn-in').value),
            thinning: parseInt(this.container.querySelector('#thinning').value),
            randomSeed: this.container.querySelector('#random-seed').value ?
                parseInt(this.container.querySelector('#random-seed').value) : null,
            cacheResults: this.container.querySelector('#cache-results').checked
        };
    }
    async generateSamples() {
        if (this.isGenerating)
            return;
        this.isGenerating = true;
        this.updateStatus('Generating...');
        this.container.querySelector('#generate-button').disabled = true;
        const startTime = performance.now();
        const settings = this.getSettings();
        try {
            const result = await this.onSamplingRequest(settings);
            this.updateStats(result, performance.now() - startTime);
        }
        catch (error) {
            console.error('Sampling failed:', error);
            this.updateStatus(`Error: ${error instanceof Error ? error.message : String(error)}`);
        }
        finally {
            this.isGenerating = false;
            this.container.querySelector('#generate-button').disabled = false;
            this.updateStatus('Idle');
        }
    }
    updateStats(result, timeMs) {
        const { totalSamples = 0, acceptedSamples = 0, rejectedSamples = 0 } = result;
        const acceptedPercentage = totalSamples ?
            ((acceptedSamples / totalSamples) * 100).toFixed(1) : '0';
        const rejectedPercentage = totalSamples ?
            ((rejectedSamples / totalSamples) * 100).toFixed(1) : '0';
        this.container.querySelector('#stat-total').textContent = totalSamples.toString();
        this.container.querySelector('#stat-accepted').textContent =
            `${acceptedSamples} (${acceptedPercentage}%)`;
        this.container.querySelector('#stat-rejected').textContent =
            `${rejectedSamples} (${rejectedPercentage}%)`;
        this.container.querySelector('#stat-time').textContent =
            `${(timeMs / 1000).toFixed(2)}s`;
        this.lastUpdateTime = new Date();
        this.updateLastUpdateTime();
    }
    updateStatus(status) {
        const statusText = this.container.querySelector('#status-text');
        statusText.textContent = `Status: ${status}`;
        statusText.classList.toggle('generating', status === 'Generating...');
        this.updateLastUpdateTime();
    }
    updateLastUpdateTime() {
        const lastUpdateEl = this.container.querySelector('#last-update');
        if (this.lastUpdateTime) {
            const timeStr = this.lastUpdateTime.toLocaleTimeString();
            lastUpdateEl.textContent = `Last Updated: ${timeStr}`;
        }
        else {
            lastUpdateEl.textContent = '';
        }
    }
}
//# sourceMappingURL=sampling-controls.js.map