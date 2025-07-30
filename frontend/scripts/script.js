// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const difficultySlider = document.getElementById('difficultySlider');
const difficultyValue = document.getElementById('difficultyValue');
const generateBtn = document.getElementById('generateBtn');
const customInstructions = document.getElementById('customInstructions');
const previewContent = document.querySelector('.preview-content');

// File handling
let uploadedFiles = [];

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    initializeSliders();
    initializeCheckboxes();
    updatePreview();
});

function initializeEventListeners() {
    // File upload events
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop events
    dropZone.addEventListener('dragover', handleDragOver);
    dropZone.addEventListener('drop', handleDrop);
    dropZone.addEventListener('dragleave', handleDragLeave);
    dropZone.addEventListener('click', () => fileInput.click());
    
    // Generate button
    generateBtn.addEventListener('click', handleGenerate);
    
    // Question type checkboxes
    const checkboxes = document.querySelectorAll('input[name="questionType"]');
    checkboxes.forEach(checkbox => {
        checkbox.addEventListener('change', updatePreview);
    });
}

function initializeSliders() {
    // Difficulty slider
    difficultySlider.addEventListener('input', function() {
        difficultyValue.textContent = this.value;
        updateSliderPosition(this, difficultyValue);
    });
    
    // Set initial position
    updateSliderPosition(difficultySlider, difficultyValue);
}

function updateSliderPosition(slider, valueElement) {
    const percent = ((slider.value - slider.min) / (slider.max - slider.min)) * 100;
    valueElement.style.left = `${percent}%`;
}

function initializeCheckboxes() {
    const checkboxItems = document.querySelectorAll('.checkbox-item');
    checkboxItems.forEach(item => {
        const checkbox = item.querySelector('input[type="checkbox"]');
        
        item.addEventListener('click', function(e) {
            if (e.target !== checkbox) {
                checkbox.checked = !checkbox.checked;
                checkbox.dispatchEvent(new Event('change'));
            }
        });
        
        checkbox.addEventListener('change', function() {
            if (this.checked) {
                item.classList.add('checked');
            } else {
                item.classList.remove('checked');
            }
        });
        
        // Set initial state
        if (checkbox.checked) {
            item.classList.add('checked');
        }
    });
}

function handleDragOver(e) {
    e.preventDefault();
    dropZone.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    dropZone.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    
    const files = Array.from(e.dataTransfer.files);
    processFiles(files);
}

function handleFileSelect(e) {
    const files = Array.from(e.target.files);
    processFiles(files);
}

function processFiles(files) {
    const validFiles = files.filter(file => {
        const validTypes = ['application/pdf', 'application/msword', 
                           'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 
                           'text/plain'];
        return validTypes.includes(file.type);
    });
    
    if (validFiles.length === 0) {
        alert('Please upload valid document files (PDF, DOC, DOCX, TXT)');
        return;
    }
    
    uploadedFiles = [...uploadedFiles, ...validFiles];
    displayUploadedFiles();
    updateUploadArea();
}

function displayUploadedFiles() {
    const existingList = document.querySelector('.file-list');
    if (existingList) {
        existingList.remove();
    }
    
    if (uploadedFiles.length === 0) return;
    
    const fileList = document.createElement('div');
    fileList.className = 'file-list';
    
    uploadedFiles.forEach((file, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        
        fileItem.innerHTML = `
            <span>📄</span>
            <span class="file-name">${file.name}</span>
            <button class="remove-btn" onclick="removeFile(${index})">×</button>
        `;
        
        fileList.appendChild(fileItem);
    });
    
    document.querySelector('.upload-area').appendChild(fileList);
}

function removeFile(index) {
    uploadedFiles.splice(index, 1);
    displayUploadedFiles();
    updateUploadArea();
}

function updateUploadArea() {
    const dropZoneContent = dropZone.querySelector('p');
    if (uploadedFiles.length > 0) {
        dropZone.style.borderStyle = 'solid';
        dropZone.style.borderColor = '#28a745';
        dropZone.style.background = '#f8fff9';
    } else {
        dropZone.style.borderStyle = 'dashed';
        dropZone.style.borderColor = '#ddd';
        dropZone.style.background = '#fafafa';
    }
}

function updatePreview() {
    const selectedTypes = Array.from(document.querySelectorAll('input[name="questionType"]:checked'))
        .map(cb => cb.value);
    
    if (selectedTypes.length === 0) {
        previewContent.innerHTML = '<p>Select question types to see sample questions here.</p>';
        return;
    }
    
    const sampleQuestions = generateSampleQuestions(selectedTypes);
    previewContent.innerHTML = `
        <h4>Sample Questions (${selectedTypes.length} types selected):</h4>
        <ul style="margin-top: 10px; padding-left: 20px;">
            ${sampleQuestions.map(q => `<li style="margin-bottom: 8px;">${q}</li>`).join('')}
        </ul>
    `;
}

function generateSampleQuestions(types) {
    const samples = {
        character: "Who is the main character in this story?",
        setting: "Where does the story take place?",
        action: "What happened after the character made their decision?",
        causal: "Why did the character choose this particular course of action?",
        outcome: "How was the conflict resolved at the end?",
        feeling: "How did the protagonist feel about the outcome?",
        prediction: "What do you think will happen next?",
        analytical: "What themes are explored in this text?",
        factual: "What are the key facts presented in the document?",
        comparative: "How does this approach compare to alternative methods?"
    };
    
    return types.map(type => samples[type] || "Sample question for " + type);
}

async function handleGenerate() {
    const selectedTypes = Array.from(document.querySelectorAll('input[name="questionType"]:checked'))
        .map(cb => cb.value);
    
    if (uploadedFiles.length === 0) {
        alert('Please upload at least one document first.');
        return;
    }
    
    if (selectedTypes.length === 0) {
        alert('Please select at least one question type.');
        return;
    }
    
    // Show loading state
    generateBtn.textContent = '🔄 Generating Questions...';
    generateBtn.disabled = true;
    
    // Show loading in preview
    previewContent.innerHTML = `
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 24px; margin-bottom: 10px;">⏳</div>
            <p><strong>Generating questions...</strong></p>
            <p style="color: #666; font-size: 0.9rem;">This may take a few moments depending on document size.</p>
        </div>
    `;
    
    try {
        // Prepare form data for the API call
        const formData = new FormData();
        
        // Add uploaded files
        uploadedFiles.forEach(file => {
            formData.append('files', file);
        });
        
        // Add configuration as JSON
        const config = {
            questionTypes: selectedTypes,
            difficulty: parseInt(difficultySlider.value),
            customInstructions: customInstructions.value
        };
        
        formData.append('config', JSON.stringify(config));
        
        // Call the backend API
        const response = await fetch('http://convai-srv-01.cs.illinois.edu:5001/process', {
            method: 'POST',
            body: formData,
            // Don't set Content-Type header - let browser set it with boundary for FormData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        // Update the preview with the generated questions and answers
        updatePreviewWithGeneratedQuestions(result.questions, result.answers);
        
        console.log('Generation completed successfully:', result);
        
    } catch (error) {
        console.error('Error generating questions:', error);
        alert(`Error generating questions: ${error.message}`);
        
        // Reset preview to show error state
        previewContent.innerHTML = `
            <div style="color: #dc3545; padding: 10px; background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 4px;">
                <strong>Error:</strong> Failed to generate questions. Please try again.
            </div>
        `;
    } finally {
        // Reset button state
        generateBtn.textContent = '🔄 Generate Questions';
        generateBtn.disabled = false;
    }
}

// Function to update preview with generated questions and answers
function updatePreviewWithGeneratedQuestions(questions, answers = []) {
    if (!questions || questions.length === 0) {
        previewContent.innerHTML = '<p>No questions were generated.</p>';
        return;
    }
    
    const questionsHtml = questions.map((question, index) => {
        let questionText = '';
        let answerText = '';
        
        // Handle different question formats
        if (typeof question === 'string') {
            questionText = question;
        } else if (question.question) {
            questionText = question.question;
        } else {
            questionText = JSON.stringify(question);
        }
        
        // Handle answer if available
        if (answers && answers[index]) {
            if (typeof answers[index] === 'string') {
                answerText = answers[index];
            } else if (answers[index].answer) {
                answerText = answers[index].answer;
            } else {
                answerText = JSON.stringify(answers[index]);
            }
        }
        
        return `
            <li style="margin-bottom: 16px; border-bottom: 1px solid #eee; padding-bottom: 12px;">
                <div style="margin-bottom: 8px;">
                    <strong style="color: #2c3e50;">Q${index + 1}:</strong> 
                    <span style="margin-left: 5px;">${questionText}</span>
                </div>
                ${answerText ? `
                    <div style="margin-left: 20px; padding: 8px; background: #f8f9fa; border-left: 3px solid #28a745; border-radius: 4px;">
                        <strong style="color: #28a745; font-size: 0.9rem;">Answer:</strong> 
                        <span style="margin-left: 5px; color: #555;">${answerText}</span>
                    </div>
                ` : ''}
            </li>
        `;
    }).join('');
    
    const totalWithAnswers = answers ? answers.filter(a => a && a !== "No answer provided").length : 0;
    
    previewContent.innerHTML = `
        <h4>🎉 Generated Questions & Answers (${questions.length} questions${answers && answers.length > 0 ? `, ${totalWithAnswers} with answers` : ''}):</h4>
        <ul style="margin-top: 15px; padding-left: 20px; max-height: 500px; overflow-y: auto; list-style: none;">
            ${questionsHtml}
        </ul>
        <div style="margin-top: 15px; padding: 12px; background: #d4edda; border: 1px solid #c3e6cb; border-radius: 4px; color: #155724;">
            <strong>✅ Success!</strong> Questions${answers && answers.length > 0 ? ' and answers' : ''} have been generated successfully. You can now review them above.
        </div>
    `;
}

// Global function for removing files (called from HTML)
window.removeFile = removeFile;
