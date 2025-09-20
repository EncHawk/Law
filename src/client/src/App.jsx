import React, { useState, useEffect } from 'react';
import { Upload, FileText, AlertTriangle, HelpCircle, CheckCircle, X, Loader2, Send, Shield, AlertCircle } from 'lucide-react';

const API_URL = import.meta.env.PROD
  ? 'https://law-nhjj.onrender.com'  // Your actual Render URL
  : import.meta.env.DEV_API_URL;

const App = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadedFileName, setUploadedFileName] = useState('');
  const [uploadStatus, setUploadStatus] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [sources, setSources] = useState([]);
  const [isAsking, setIsAsking] = useState(false);
  const [risks, setRisks] = useState([]);
  const [isScanning, setIsScanning] = useState(false);
  const [error, setError] = useState('');
  const [documents, setDocuments] = useState([]);
  const [activeTab, setActiveTab] = useState('upload');

  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const response = await fetch(`${API_URL}/documents`);
      const data = await response.json();
      setDocuments(data.documents || []);
    } catch (err) {
      console.error('Error fetching documents:', err);
    }
  };

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      const allowedTypes = ['.pdf', '.docx', '.doc', '.txt'];
      const fileExtension = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
      
      if (!allowedTypes.includes(fileExtension)) {
        setError(`Invalid file type. Allowed types: ${allowedTypes.join(', ')}`);
        return;
      }
      
      if (file.size > 10 * 1024 * 1024) { // 10MB limit
        setError('File size must be less than 10MB');
        return;
      }
      
      setSelectedFile(file);
      setError('');
      setUploadStatus('');
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a file first');
      return;
    }

    setIsUploading(true);
    setError('');
    setUploadStatus('');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`${API_URL}/upload`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setUploadedFileName(data.filename);
        setUploadStatus(data.message);
        setSelectedFile(null);
        setActiveTab('ask');
        fetchDocuments();
        // Clear file input
        const fileInput = document.getElementById('file-input');
        if (fileInput) fileInput.value = '';
      } else {
        setError(data.detail || 'Upload failed');
      }
    } catch (err) {
      setError('Failed to upload file. Please ensure the backend is running.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleAskQuestion = async () => {
    if (!uploadedFileName) {
      setError('Please upload a document first');
      return;
    }

    if (!question.trim()) {
      setError('Please enter a question');
      return;
    }

    setIsAsking(true);
    setError('');
    setAnswer('');
    setSources([]);

    try {
      const response = await fetch(`${API_URL}/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          filename: uploadedFileName,
          question: question.trim(),
        }),
      });

      const data = await response.json();

      if (response.ok) {
        setAnswer(data.answer);
        setSources(data.sources || []);
      } else {
        setError(data.detail || 'Failed to get answer');
      }
    } catch (err) {
      setError('Failed to get answer. Please ensure the backend is running.');
    } finally {
      setIsAsking(false);
    }
  };

  const handleScanForRisks = async () => {
  if (!uploadedFileName) {
    setError('Please upload a document first');
    return;
  }

  setIsScanning(true);
  setError('');
  setRisks([]);

  try {
    console.log('Scanning document:', uploadedFileName); // Debug log
    
    const response = await fetch(`${API_URL}/scan`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        filename: uploadedFileName,
      }),
    });

    console.log('Response status:', response.status); // Debug log
    
    const data = await response.json();
    console.log('Response data:', data); // Debug log

    if (response.ok) {
      // More defensive parsing of the response
      const risksArray = data.risks || data.risk_clauses || [];
      console.log('Parsed risks array:', risksArray); // Debug log
      
      if (Array.isArray(risksArray)) {
        // Ensure each risk object has all required properties
        const validRisks = risksArray.filter(risk => 
          risk && typeof risk === 'object' && risk.clause_type
        ).map(risk => ({
          clause_type: risk.clause_type || 'Unknown',
          text: risk.text || '',
          explanation: risk.explanation || 'No explanation provided',
          severity: (risk.severity || 'medium').toLowerCase()
        }));
        
        console.log('Valid risks after processing:', validRisks); // Debug log
        setRisks(validRisks);
        
        if (validRisks.length > 0) {
          setActiveTab('risks');
        } else {
          setError('No risks found in the document');
        }
      } else {
        console.error('Risks data is not an array:', risksArray);
        setError('Invalid response format: risks should be an array');
      }
    } else {
      console.error('API error:', data);
      setError(data.detail || 'Failed to scan document');
    }
  } catch (err) {
    console.error('Scan error:', err);
    setError('Failed to scan document. Please ensure the backend is running.');
  } finally {
    setIsScanning(false);
  }
};

  const handleSelectDocument = (filename) => {
    setUploadedFileName(filename);
    setActiveTab('ask');
    setError('');
    setAnswer('');
    setRisks([]);
  };

  const getSeverityColor = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'high':
        return 'bg-red-100 text-red-800 border-red-200';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'low':
        return 'bg-blue-100 text-blue-800 border-blue-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getSeverityIcon = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'high':
        return <AlertTriangle className="w-5 h-5" />;
      case 'medium':
        return <AlertCircle className="w-5 h-5" />;
      default:
        return <Shield className="w-5 h-5" />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <FileText className="w-8 h-8 text-indigo-600" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">ClauseBuddy</h1>
                <p className="text-sm text-gray-600">Say NO hidden clauses! TRY NOW.</p>
              </div>
            </div>
            {uploadedFileName && (
              <div className="flex items-center space-x-2 bg-green-50 px-3 py-1 rounded-full">
                <CheckCircle className="w-4 h-4 text-green-600" />
                <span className="text-sm text-green-800 font-medium">{uploadedFileName}</span>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
        {/* Error Alert */}
        {error && (
          <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4 flex items-start space-x-3">
            <AlertTriangle className="w-5 h-5 text-red-600 mt-0.5" />
            <div className="flex-1">
              <p className="text-sm text-red-800">{error}</p>
            </div>
            <button
              onClick={() => setError('')}
              className="text-red-600 hover:text-red-800"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        )}

        {/* Success Alert */}
        {uploadStatus && (
          <div className="mb-6 bg-green-50 border border-green-200 rounded-lg p-4 flex items-start space-x-3">
            <CheckCircle className="w-5 h-5 text-green-600 mt-0.5" />
            <div className="flex-1">
              <p className="text-sm text-green-800">{uploadStatus}</p>
            </div>
          </div>
        )}

        {/* Tab Navigation */}
        <div className="mb-8 border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
            <button
              onClick={() => setActiveTab('upload')}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'upload'
                  ? 'border-indigo-500 text-indigo-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Upload Document
            </button>
            <button
              onClick={() => setActiveTab('ask')}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'ask'
                  ? 'border-indigo-500 text-indigo-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              } ${!uploadedFileName && 'opacity-50 cursor-not-allowed'}`}
              disabled={!uploadedFileName}
            >
              Ask Questions
            </button>
            <button
              onClick={() => setActiveTab('risks')}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'risks'
                  ? 'border-indigo-500 text-indigo-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              } ${!uploadedFileName && 'opacity-50 cursor-not-allowed'}`}
              disabled={!uploadedFileName}
            >
              Risk Analysis
            </button>
          </nav>
        </div>

        {/* Upload Section */}
        {activeTab === 'upload' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <Upload className="w-5 h-5 mr-2 text-indigo-600" />
                Upload Legal Document
              </h2>
              
              <div className="space-y-4">
                <div
                  className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-indigo-400 transition-colors cursor-pointer"
                  onClick={() => document.getElementById('file-input').click()}
                >
                  <Upload className="w-12 h-12 mx-auto text-gray-400 mb-3" />
                  <p className="text-sm text-gray-600 mb-2">
                    Click to upload or drag and drop
                  </p>
                  <p className="text-xs text-gray-500">
                    PDF, DOCX, DOC, TXT (max 10MB)
                  </p>
                  <input
                    id="file-input"
                    type="file"
                    accept=".pdf,.docx,.doc,.txt"
                    onChange={handleFileSelect}
                    className="hidden"
                  />
                </div>

                {selectedFile && (
                  <div className="bg-gray-50 rounded-lg p-3 flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <FileText className="w-5 h-5 text-gray-600" />
                      <span className="text-sm text-gray-900">{selectedFile.name}</span>
                      <span className="text-xs text-gray-500">
                        ({(selectedFile.size / 1024).toFixed(1)} KB)
                      </span>
                    </div>
                    <button
                      onClick={() => setSelectedFile(null)}
                      className="text-gray-400 hover:text-gray-600"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                )}

                <button
                  onClick={handleUpload}
                  disabled={!selectedFile || isUploading}
                  className="w-full bg-indigo-600 text-white py-2 px-4 rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center"
                >
                  {isUploading ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Uploading...
                    </>
                  ) : (
                    <>
                      <Upload className="w-4 h-4 mr-2" />
                      Upload Document
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Recent Documents */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Recent Documents</h2>
              {documents.length === 0 ? (
                <p className="text-gray-500 text-sm">No documents uploaded yet</p>
              ) : (
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {documents.map((doc) => (
                    <div
                      key={doc.filename}
                      onClick={() => handleSelectDocument(doc.filename)}
                      className="p-3 bg-gray-50 rounded-lg hover:bg-gray-100 cursor-pointer transition-colors flex items-center justify-between group"
                    >
                      <div className="flex items-center space-x-3">
                        <FileText className="w-4 h-4 text-gray-500" />
                        <div>
                          <p className="text-sm font-medium text-gray-900">{doc.filename}</p>
                          <p className="text-xs text-gray-500">
                            {(doc.size / 1024).toFixed(1)} KB • {new Date(doc.uploaded_at).toLocaleDateString()}
                          </p>
                        </div>
                      </div>
                      <span className="text-xs text-indigo-600 opacity-0 group-hover:opacity-100 transition-opacity">
                        Select →
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Ask Questions Section */}
        {activeTab === 'ask' && (
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <HelpCircle className="w-5 h-5 mr-2 text-indigo-600" />
              Ask Questions About Your Document
            </h2>

            {!uploadedFileName ? (
              <div className="text-center py-12">
                <FileText className="w-16 h-16 mx-auto text-gray-300 mb-4" />
                <p className="text-gray-500">Please upload a document first to ask questions</p>
              </div>
            ) : (
              <div className="space-y-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Your Question
                  </label>
                  <div className="flex space-x-3">
                    <input
                      type="text"
                      value={question}
                      onChange={(e) => setQuestion(e.target.value)}
                      onKeyPress={(e) => e.key === 'Enter' && handleAskQuestion()}
                      placeholder="e.g., What are my payment obligations? What happens if I terminate early?"
                      className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none"
                    />
                    <button
                      onClick={handleAskQuestion}
                      disabled={!question.trim() || isAsking}
                      className="bg-indigo-600 text-white px-6 py-2 rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center"
                    >
                      {isAsking ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <Send className="w-4 h-4" />
                      )}
                    </button>
                  </div>
                </div>

                {/* Sample Questions */}
                <div className="bg-gray-50 rounded-lg p-4">
                  <p className="text-xs font-medium text-gray-600 mb-2">Sample questions:</p>
                  <div className="flex flex-wrap gap-2">
                    {[
                      "What is the termination period?",
                      "Are there any penalties?",
                      "What are my obligations?",
                      "Is arbitration required?"
                    ].map((q) => (
                      <button
                        key={q}
                        onClick={() => setQuestion(q)}
                        className="text-xs bg-white px-3 py-1 rounded-full text-gray-700 hover:bg-indigo-50 hover:text-indigo-700 transition-colors"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Answer Display */}
                {answer && (
                  <div className="bg-indigo-50 rounded-lg p-4 space-y-4">
                    <div>
                      <h3 className="font-medium text-gray-900 mb-2">Answer:</h3>
                      <p className="text-gray-800 leading-relaxed">{answer}</p>
                    </div>
                    
                    {sources.length > 0 && (
                      <div>
                        <h4 className="text-sm font-medium text-gray-700 mb-2">Sources from document:</h4>
                        <div className="space-y-2">
                          {sources.map((source, idx) => (
                            <div key={idx} className="text-xs text-gray-600 bg-white p-2 rounded border border-indigo-100">
                              "{source}"
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Risk Analysis Section */}
        {activeTab === 'risks' && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-lg font-semibold text-gray-900 flex items-center">
              <AlertTriangle className="w-5 h-5 mr-2 text-indigo-600" />
              Risk Analysis
            </h2>
            {uploadedFileName && (
              <button
                onClick={handleScanForRisks}
                disabled={isScanning}
                className="bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center text-sm"
              >
                {isScanning ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Scanning...
                  </>
                ) : (
                  <>
                    <Shield className="w-4 h-4 mr-2" />
                    Scan for Risks
                  </>
                )}
              </button>
            )}
          </div>

          {!uploadedFileName ? (
            <div className="text-center py-12">
              <AlertTriangle className="w-16 h-16 mx-auto text-gray-300 mb-4" />
              <p className="text-gray-500">Please upload a document first to scan for risks</p>
            </div>
          ) : risks.length === 0 && !isScanning ? (
            <div className="text-center py-12">
              <Shield className="w-16 h-16 mx-auto text-gray-300 mb-4" />
              <p className="text-gray-500">
                Click "Scan for Risks" to identify important clauses in your document
              </p>
            </div>
          ) : isScanning ? (
            <div className="text-center py-12">
              <Loader2 className="w-16 h-16 mx-auto text-indigo-300 mb-4 animate-spin" />
              <p className="text-gray-500">Analyzing document for risks...</p>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="bg-blue-50 rounded-lg p-3 mb-4">
                <p className="text-sm text-blue-800">
                  Found <span className="font-semibold">{risks.length}</span> important clause{risks.length !== 1 ? 's' : ''} to review
                </p>
              </div>
              
              {risks.map((risk, idx) => (
                <div
                  key={`${risk.clause_type}-${idx}`}
                  className={`border rounded-lg p-4 space-y-3 ${getSeverityColor(risk.severity)}`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-center space-x-2">
                      {getSeverityIcon(risk.severity)}
                      <h3 className="font-medium">{risk.clause_type || 'Unknown Clause'}</h3>
                    </div>
                    <span className={`text-xs px-2 py-1 rounded-full font-medium ${
                      risk.severity === 'high' ? 'bg-red-200' :
                      risk.severity === 'medium' ? 'bg-yellow-200' :
                      'bg-blue-200'
                    }`}>
                      {(risk.severity || 'medium').toUpperCase()}
                    </span>
                  </div>
                  
                  {risk.text && risk.text.trim() && (
                    <div className="bg-white bg-opacity-50 rounded p-2">
                      <p className="text-xs text-gray-700 italic">"{risk.text}"</p>
                    </div>
                  )}
                  
                  <div>
                    <p className="text-sm font-medium mb-1">What this means:</p>
                    <p className="text-sm leading-relaxed">
                      {risk.explanation || 'No explanation available'}
                    </p>
                  </div>
                </div>
              ))}
              
              {/* Debug information (remove in production) */}
              {process.env.NODE_ENV === 'development' && (
                <div className="mt-8 p-4 bg-gray-100 rounded-lg">
                  <details>
                    <summary className="text-sm font-medium cursor-pointer">Debug Info</summary>
                    <pre className="mt-2 text-xs overflow-auto">
                      {JSON.stringify(risks, null, 2)}
                    </pre>
                  </details>
                </div>
              )}
            </div>
          )}
        </div>
      )}
      </main>

      {/* Footer */}
      <footer className="mt-16 border-t border-gray-200 bg-white">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <p className="text-center text-sm text-gray-500">
            ClauseBuddy, your one stop solution for all contracts.
            made by <a href="https://x.com/d_leap07" target="_blank">Dilip Kumar R</a>
          </p>
        </div>
      </footer>
    </div>
  );
};

export default App;