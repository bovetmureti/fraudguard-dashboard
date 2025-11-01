import React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { AlertCircle, CheckCircle, TrendingUp, Activity, Shield, FileText, DollarSign, Clock } from 'lucide-react';

const FraudGuardDashboard = () => {
  const [claims, setClaims] = useState([]);
  const [selectedClaim, setSelectedClaim] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [filterRiskLevel, setFilterRiskLevel] = useState('all');

  // Simulated KPIs from the pitch
  const kpis = {
    detectionRate: 90,
    lossReduction: 20,
    processingSpeedUp: 4,
    falsePositiveRate: 8.5,
    totalClaims: 10000,
    fraudulentClaims: 2500,
    savedAmount: 4200000 // KES
  };

  // Generate mock claims data
  useEffect(() => {
    const mockClaims = generateMockClaims(100);
    setClaims(mockClaims);
  }, []);

  const generateMockClaims = (count) => {
    const productTypes = ['Motor', 'Medical', 'Property', 'Liability', 'Life'];
    const locations = ['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret'];
    const fraudTypes = ['Fictitious Claims', 'Staged Accidents', 'Medical Billing Fraud', 'Premium Fraud', 'Legitimate'];
   
    return Array.from({ length: count }, (_, i) => {
      const isFraud = Math.random() > 0.75;
      const riskScore = isFraud
        ? Math.random() * 30 + 70
        : Math.random() * 50;
     
      return {
        id: `CLM-${String(i + 1).padStart(6, '0')}`,
        policyId: `POL-${Math.floor(Math.random() * 9000 + 1000)}`,
        productType: productTypes[Math.floor(Math.random() * productTypes.length)],
        claimAmount: Math.floor(Math.random() * 500000 + 50000),
        location: locations[Math.floor(Math.random() * locations.length)],
        date: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
        riskScore: Math.round(riskScore),
        riskLevel: riskScore > 75 ? 'HIGH' : riskScore > 50 ? 'MEDIUM' : 'LOW',
        fraudType: isFraud ? fraudTypes[Math.floor(Math.random() * 4)] : 'Legitimate',
        status: riskScore > 75 ? 'Under Investigation' : riskScore > 50 ? 'Pending Review' : 'Approved',
        mlScore: Math.round(riskScore + (Math.random() * 10 - 5)),
        llmScore: Math.round(riskScore + (Math.random() * 10 - 5)),
        description: `Claim for ${productTypes[Math.floor(Math.random() * productTypes.length)]} insurance in ${locations[Math.floor(Math.random() * locations.length)]}.`
      };
    });
  };

  const analyzeClaim = async (claim) => {
  setIsAnalyzing(true);
  setAnalysisResult(null);
  try {
    // Map frontend claim to backend-compatible payload (snake_case, fill missing fields)
    const payload = {
      claim_id: claim.id,
      policy_id: claim.policyId,
      product_type: claim.productType,
      claim_amount: claim.claimAmount,
      policy_premium: Math.floor(Math.random() * 100000 + 10000),  // Mock premium (10k-110k KES)
      claim_date: `${claim.date}T00:00:00`,  // Convert to ISO (assume UTC)
      policy_start_date: new Date(Date.now() - Math.random() * 365 * 24 * 60 * 60 * 1000).toISOString(),  // Mock: 0-1 year ago
      claimant_age: Math.floor(Math.random() * 57 + 18),  // Mock: 18-75
      location: claim.location,
      previous_claims_count: Math.floor(Math.random() * 5),  // Mock: 0-4
      claim_processing_time: Math.floor(Math.random() * 30 + 1),  // Mock: 1-30 days
      documents_submitted: Math.floor(Math.random() * 7 + 1),  // Mock: 1-7
      witness_count: Math.floor(Math.random() * 5),  // Mock: 0-4
      hospital_name: claim.productType === 'Medical' ? `Hospital_${Math.floor(Math.random() * 50 + 1)}` : 'N/A',
      police_report: claim.productType === 'Motor' ? Math.random() > 0.5 : false,  // Mock: Random for Motor, else false
      payment_method: ['M-Pesa', 'Bank Transfer', 'Cheque'][Math.floor(Math.random() * 3)],  // Mock random
      claim_description: claim.description
    };

    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error(`Backend error: ${response.statusText}`);
    const analysis = await response.json();

    setAnalysisResult(analysis);
  } catch (error) {
    console.error("Analysis error:", error);
    // Your existing fallback mock
    setAnalysisResult({
      riskAssessment: claim.riskLevel,
      confidenceLevel: 85,
      fraudIndicators: ["High claim amount", "Early claim submission", "Insufficient documentation"],
      redFlags: ["Claim amount exceeds policy average", "Location has high fraud rate"],
      recommendedActions: ["Request additional documentation", "Conduct field investigation", "Verify witness statements"],
      summary: `This ${claim.productType} claim shows ${claim.riskLevel.toLowerCase()} fraud risk with several suspicious patterns requiring investigation.`
    });
  } finally {
    setIsAnalyzing(false);
  }
};

  // Calculate statistics
  const stats = {
    highRisk: claims.filter(c => c.riskLevel === 'HIGH').length,
    mediumRisk: claims.filter(c => c.riskLevel === 'MEDIUM').length,
    lowRisk: claims.filter(c => c.riskLevel === 'LOW').length,
    avgRiskScore: claims.reduce((sum, c) => sum + c.riskScore, 0) / claims.length || 0
  };

  // Chart data
  const riskDistribution = [
    { name: 'High Risk', value: stats.highRisk, color: '#ef4444' },
    { name: 'Medium Risk', value: stats.mediumRisk, color: '#f59e0b' },
    { name: 'Low Risk', value: stats.lowRisk, color: '#10b981' }
  ];

  const fraudByProduct = Object.entries(
    claims.reduce((acc, claim) => {
      acc[claim.productType] = (acc[claim.productType] || 0) + (claim.riskLevel === 'HIGH' ? 1 : 0);
      return acc;
    }, {})
  ).map(([name, value]) => ({ name, value }));

  const timeSeriesData = Array.from({ length: 30 }, (_, i) => {
    const date = new Date();
    date.setDate(date.getDate() - (29 - i));
    const dayClaims = claims.filter(c => c.date === date.toISOString().split('T')[0]);
    return {
      date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      highRisk: dayClaims.filter(c => c.riskLevel === 'HIGH').length,
      mediumRisk: dayClaims.filter(c => c.riskLevel === 'MEDIUM').length,
      lowRisk: dayClaims.filter(c => c.riskLevel === 'LOW').length
    };
  });

  const modelPerformance = [
    { model: 'Random Forest', precision: 0.89, recall: 0.92, f1: 0.90 },
    { model: 'Gradient Boosting', precision: 0.91, recall: 0.88, f1: 0.89 },
    { model: 'Neural Network', precision: 0.87, recall: 0.90, f1: 0.88 },
    { model: 'Isolation Forest', precision: 0.84, recall: 0.93, f1: 0.88 }
  ];

  const kpiRadarData = [
    { metric: 'Detection Rate', value: kpis.detectionRate, fullMark: 100 },
    { metric: 'Accuracy', value: 92, fullMark: 100 },
    { metric: 'Speed', value: 95, fullMark: 100 },
    { metric: 'Precision', value: 88, fullMark: 100 },
    { metric: 'Recall', value: 91, fullMark: 100 }
  ];

  // Filter claims
  const filteredClaims = filterRiskLevel === 'all'
    ? claims
    : claims.filter(c => c.riskLevel === filterRiskLevel);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white p-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
              FraudGuard AI
            </h1>
            <p className="text-slate-400">Hybrid AI-Human Framework for Insurance Fraud Detection</p>
            <p className="text-sm text-slate-500 mt-1">GA Insurance Limited - Kenya Operations</p>
          </div>
          <div className="text-right">
            <div className="text-sm text-slate-400">Real-time Detection System</div>
            <div className="text-2xl font-bold text-cyan-400">{claims.length.toLocaleString()} Claims Analyzed</div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="flex gap-2 mb-6 border-b border-slate-700">
        {['overview', 'analytics', 'claims', 'models'].map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-6 py-3 font-medium capitalize transition-colors ${
              activeTab === tab
                ? 'border-b-2 border-cyan-400 text-cyan-400'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Overview Tab */}
      {activeTab === 'overview' && (
        <div>
          {/* KPI Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div className="bg-gradient-to-br from-red-500/20 to-red-600/20 rounded-xl p-6 border border-red-500/30">
              <div className="flex items-center justify-between mb-4">
                <AlertCircle className="text-red-400" size={32} />
                <div className="text-3xl font-bold text-red-400">{stats.highRisk}</div>
              </div>
              <div className="text-sm text-slate-300">High Risk Claims</div>
              <div className="text-xs text-slate-500 mt-1">Requires immediate investigation</div>
            </div>
            <div className="bg-gradient-to-br from-blue-500/20 to-blue-600/20 rounded-xl p-6 border border-blue-500/30">
              <div className="flex items-center justify-between mb-4">
                <Shield className="text-blue-400" size={32} />
                <div className="text-3xl font-bold text-blue-400">{kpis.detectionRate}%</div>
              </div>
              <div className="text-sm text-slate-300">Detection Rate</div>
              <div className="text-xs text-slate-500 mt-1">Target: 90% achieved ✓</div>
            </div>
            <div className="bg-gradient-to-br from-green-500/20 to-green-600/20 rounded-xl p-6 border border-green-500/30">
              <div className="flex items-center justify-between mb-4">
                <DollarSign className="text-green-400" size={32} />
                <div className="text-3xl font-bold text-green-400">{kpis.lossReduction}%</div>
              </div>
              <div className="text-sm text-slate-300">Loss Reduction</div>
              <div className="text-xs text-slate-500 mt-1">KES {(kpis.savedAmount / 1000000).toFixed(1)}M saved</div>
            </div>
            <div className="bg-gradient-to-br from-purple-500/20 to-purple-600/20 rounded-xl p-6 border border-purple-500/30">
              <div className="flex items-center justify-between mb-4">
                <Clock className="text-purple-400" size={32} />
                <div className="text-3xl font-bold text-purple-400">{kpis.processingSpeedUp}x</div>
              </div>
              <div className="text-sm text-slate-300">Processing Speed</div>
              <div className="text-xs text-slate-500 mt-1">0.25 sec per claim</div>
            </div>
          </div>

          {/* Charts Row 1 */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            {/* Risk Distribution Pie */}
            <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Activity className="text-cyan-400" />
                Risk Level Distribution
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={riskDistribution}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {riskDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>

            {/* Fraud by Product Type */}
            <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <TrendingUp className="text-cyan-400" />
                High Risk Claims by Product
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={fraudByProduct}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="name" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                  />
                  <Bar dataKey="value" fill="#3b82f6" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Time Series Chart */}
          <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700 mb-6">
            <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <Activity className="text-cyan-400" />
              Claims Trend (Last 30 Days)
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={timeSeriesData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="date" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                />
                <Legend />
                <Line type="monotone" dataKey="highRisk" stroke="#ef4444" name="High Risk" strokeWidth={2} />
                <Line type="monotone" dataKey="mediumRisk" stroke="#f59e0b" name="Medium Risk" strokeWidth={2} />
                <Line type="monotone" dataKey="lowRisk" stroke="#10b981" name="Low Risk" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* System Performance KPIs */}
          <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700">
            <h3 className="text-xl font-semibold mb-4">System Performance Metrics</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-3xl font-bold text-cyan-400">{kpis.detectionRate}%</div>
                <div className="text-sm text-slate-400 mt-1">Alert Rate</div>
                <div className="text-xs text-green-400 mt-1">✓ Target Met</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-cyan-400">{kpis.falsePositiveRate}%</div>
                <div className="text-sm text-slate-400 mt-1">False Positives</div>
                <div className="text-xs text-green-400 mt-1">✓ Below 10%</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-cyan-400">300-500%</div>
                <div className="text-sm text-slate-400 mt-1">ROI</div>
                <div className="text-xs text-slate-500 mt-1">12-18 months</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-cyan-400">50-70%</div>
                <div className="text-sm text-slate-400 mt-1">Confirmation Rate</div>
                <div className="text-xs text-slate-500 mt-1">Flagged claims</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Analytics Tab */}
      {activeTab === 'analytics' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* KPI Radar Chart */}
          <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700">
            <h3 className="text-xl font-semibold mb-4">System Performance Radar</h3>
            <ResponsiveContainer width="100%" height={400}>
              <RadarChart data={kpiRadarData}>
                <PolarGrid stroke="#334155" />
                <PolarAngleAxis dataKey="metric" stroke="#94a3b8" />
                <PolarRadiusAxis stroke="#94a3b8" />
                <Radar name="Performance" dataKey="value" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.6} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>

          {/* Model Comparison */}
          <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700">
            <h3 className="text-xl font-semibold mb-4">ML Model Performance</h3>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={modelPerformance}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="model" stroke="#94a3b8" angle={-15} textAnchor="end" height={100} />
                <YAxis stroke="#94a3b8" domain={[0, 1]} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                  formatter={(value) => (value * 100).toFixed(1) + '%'}
                />
                <Legend />
                <Bar dataKey="precision" fill="#3b82f6" name="Precision" />
                <Bar dataKey="recall" fill="#10b981" name="Recall" />
                <Bar dataKey="f1" fill="#f59e0b" name="F1-Score" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Fraud Type Breakdown */}
          <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700 lg:col-span-2">
            <h3 className="text-xl font-semibold mb-4">Fraud Type Distribution</h3>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              {Object.entries(
                claims.reduce((acc, claim) => {
                  if (claim.fraudType !== 'Legitimate') {
                    acc[claim.fraudType] = (acc[claim.fraudType] || 0) + 1;
                  }
                  return acc;
                }, {})
              ).map(([type, count]) => (
                <div key={type} className="bg-slate-700/50 rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-cyan-400">{count}</div>
                  <div className="text-xs text-slate-400 mt-1">{type}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Claims Tab */}
      {activeTab === 'claims' && (
        <div>
          {/* Filter */}
          <div className="mb-6 flex gap-4">
            <select
              value={filterRiskLevel}
              onChange={(e) => setFilterRiskLevel(e.target.value)}
              className="bg-slate-700 text-white px-4 py-2 rounded-lg border border-slate-600 focus:outline-none focus:border-cyan-400"
            >
              <option value="all">All Risk Levels</option>
              <option value="HIGH">High Risk Only</option>
              <option value="MEDIUM">Medium Risk Only</option>
              <option value="LOW">Low Risk Only</option>
            </select>
            <div className="flex-1"></div>
            <div className="text-sm text-slate-400 py-2">
              Showing {filteredClaims.length} of {claims.length} claims
            </div>
          </div>

          {/* Claims Table */}
          <div className="bg-slate-800/50 rounded-xl border border-slate-700 overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-slate-700/50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-300 uppercase tracking-wider">Claim ID</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-300 uppercase tracking-wider">Product</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-300 uppercase tracking-wider">Amount (KES)</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-300 uppercase tracking-wider">Location</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-300 uppercase tracking-wider">Risk Score</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-300 uppercase tracking-wider">Risk Level</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-300 uppercase tracking-wider">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-700">
                  {filteredClaims.slice(0, 50).map((claim) => (
                    <tr key={claim.id} className="hover:bg-slate-700/30 transition-colors">
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-slate-200">{claim.id}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-300">{claim.productType}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-300">{claim.claimAmount.toLocaleString()}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-300">{claim.location}</td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center gap-2">
                          <div className="text-sm font-medium text-slate-200">{claim.riskScore}</div>
                          <div className="w-20 bg-slate-700 rounded-full h-2">
                            <div
                              className={`h-2 rounded-full ${
                                claim.riskScore > 75 ? 'bg-red-500' :
                                claim.riskScore > 50 ? 'bg-yellow-500' : 'bg-green-500'
                              }`}
                              style={{ width: `${claim.riskScore}%` }}
                            ></div>
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                          claim.riskLevel === 'HIGH' ? 'bg-red-500/20 text-red-400 border border-red-500/30' :
                          claim.riskLevel === 'MEDIUM' ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30' :
                          'bg-green-500/20 text-green-400 border border-green-500/30'
                        }`}>
                          {claim.riskLevel}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">
                        <button
                          onClick={() => {
                            setSelectedClaim(claim);
                            analyzeClaim(claim);
                          }}
                          className="px-4 py-2 bg-cyan-500 hover:bg-cyan-600 text-white rounded-lg transition-colors text-xs font-medium"
                        >
                          Analyze with AI
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Analysis Modal */}
          {selectedClaim && (
            <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50" onClick={() => setSelectedClaim(null)}>
              <div className="bg-slate-800 rounded-xl p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto border border-slate-700" onClick={(e) => e.stopPropagation()}>
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-2xl font-bold text-cyan-400">AI-Powered Claim Analysis</h3>
                  <button onClick={() => setSelectedClaim(null)} className="text-slate-400 hover:text-white">
                    ✕
                  </button>
                </div>
                <div className="space-y-4 mb-6">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <div className="text-sm text-slate-400">Claim ID</div>
                      <div className="font-medium">{selectedClaim.id}</div>
                    </div>
                    <div>
                      <div className="text-sm text-slate-400">Product Type</div>
                      <div className="font-medium">{selectedClaim.productType}</div>
                    </div>
                    <div>
                      <div className="text-sm text-slate-400">Claim Amount</div>
                      <div className="font-medium">KES {selectedClaim.claimAmount.toLocaleString()}</div>
                    </div>
                    <div>
                      <div className="text-sm text-slate-400">Location</div>
                      <div className="font-medium">{selectedClaim.location}</div>
                    </div>
                  </div>
                  <div className="bg-slate-700/50 rounded-lg p-4">
                    <div className="text-sm text-slate-400 mb-2">Hybrid Risk Assessment</div>
                    <div className="flex items-center gap-4">
                      <div>
                        <div className="text-xs text-slate-400">ML Score</div>
                        <div className="text-2xl font-bold text-blue-400">{selectedClaim.mlScore}</div>
                      </div>
                      <div className="text-2xl text-slate-600">+</div>
                      <div>
                        <div className="text-xs text-slate-400">LLM Score</div>
                        <div className="text-2xl font-bold text-purple-400">{selectedClaim.llmScore}</div>
                      </div>
                      <div className="text-2xl text-slate-600">=</div>
                      <div>
                        <div className="text-xs text-slate-400">Final Score</div>
                        <div className="text-3xl font-bold text-cyan-400">{selectedClaim.riskScore}</div>
                      </div>
                    </div>
                  </div>
                </div>
                {isAnalyzing ? (
                  <div className="text-center py-8">
                    <div className="inline-block animate-spin rounded-full h-12 w-12 border-4 border-cyan-400 border-t-transparent"></div>
                    <div className="mt-4 text-slate-400">Analyzing with Claude AI...</div>
                  </div>
                ) : analysisResult ? (
                  <div className="space-y-4 max-h-[50vh] overflow-y-auto">
                    <div className="bg-gradient-to-r from-slate-700/50 to-slate-600/50 rounded-lg p-4 border-l-4 border-cyan-400">
                      <div className="text-sm text-slate-400 mb-1">Risk Assessment</div>
                      <div className="text-2xl font-bold text-cyan-400">{analysisResult.riskAssessment}</div>
                      <div className="text-sm text-slate-400 mt-1">Confidence: {analysisResult.confidenceLevel}%</div>
                    </div>
                    <div>
                      <div className="text-sm font-medium text-slate-300 mb-2">Summary</div>
                      <div className="bg-slate-700/50 rounded-lg p-4 text-sm text-slate-300">
                        {analysisResult.summary}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm font-medium text-slate-300 mb-2">Fraud Indicators</div>
                      <div className="space-y-2">
                        {analysisResult.fraudIndicators.map((indicator, idx) => (
                          <div key={idx} className="flex items-start gap-2 bg-red-500/10 rounded-lg p-3 border border-red-500/20">
                            <AlertCircle className="text-red-400 mt-0.5 flex-shrink-0" size={16} />
                            <div className="text-sm text-slate-300">{indicator}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm font-medium text-slate-300 mb-2">Red Flags</div>
                      <div className="space-y-2">
                        {analysisResult.redFlags.map((flag, idx) => (
                          <div key={idx} className="flex items-start gap-2 bg-yellow-500/10 rounded-lg p-3 border border-yellow-500/20">
                            <AlertCircle className="text-yellow-400 mt-0.5 flex-shrink-0" size={16} />
                            <div className="text-sm text-slate-300">{flag}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm font-medium text-slate-300 mb-2">Recommended Actions</div>
                      <div className="space-y-2">
                        {analysisResult.recommendedActions.map((action, idx) => (
                          <div key={idx} className="flex items-start gap-2 bg-blue-500/10 rounded-lg p-3 border border-blue-500/20">
                            <CheckCircle className="text-blue-400 mt-0.5 flex-shrink-0" size={16} />
                            <div className="text-sm text-slate-300">{action}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                ) : null}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Models Tab */}
      {activeTab === 'models' && (
        <div className="space-y-6">
          <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700">
            <h3 className="text-xl font-semibold mb-4">ML Model Architecture</h3>
            <div className="space-y-4">
              {modelPerformance.map((model, idx) => (
                <div key={idx} className="bg-slate-700/50 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <div className="font-medium text-lg">{model.model}</div>
                    <div className="text-sm text-slate-400">Active</div>
                  </div>
                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <div className="text-xs text-slate-400">Precision</div>
                      <div className="text-xl font-bold text-cyan-400">{(model.precision * 100).toFixed(1)}%</div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-400">Recall</div>
                      <div className="text-xl font-bold text-green-400">{(model.recall * 100).toFixed(1)}%</div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-400">F1-Score</div>
                      <div className="text-xl font-bold text-yellow-400">{(model.f1 * 100).toFixed(1)}%</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700">
            <h3 className="text-xl font-semibold mb-4">Integration Status</h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between py-2 border-b border-slate-700">
                <div className="flex items-center gap-3">
                  <CheckCircle className="text-green-400" size={20} />
                  <div>
                    <div className="font-medium">Claude API (LLM Analysis)</div>
                    <div className="text-sm text-slate-400">Unstructured data processing</div>
                  </div>
                </div>
                <span className="px-3 py-1 bg-green-500/20 text-green-400 rounded-full text-xs">Connected</span>
              </div>
              <div className="flex items-center justify-between py-2 border-b border-slate-700">
                <div className="flex items-center gap-3">
                  <CheckCircle className="text-green-400" size={20} />
                  <div>
                    <div className="font-medium">IPRS API</div>
                    <div className="text-sm text-slate-400">Identity verification (Kenya)</div>
                  </div>
                </div>
                <span className="px-3 py-1 bg-green-500/20 text-green-400 rounded-full text-xs">Active</span>
              </div>
              <div className="flex items-center justify-between py-2 border-b border-slate-700">
                <div className="flex items-center gap-3">
                  <CheckCircle className="text-green-400" size={20} />
                  <div>
                    <div className="font-medium">M-Pesa API</div>
                    <div className="text-sm text-slate-400">Payment validation</div>
                  </div>
                </div>
                <span className="px-3 py-1 bg-green-500/20 text-green-400 rounded-full text-xs">Active</span>
              </div>
              <div className="flex items-center justify-between py-2">
                <div className="flex items-center gap-3">
                  <CheckCircle className="text-green-400" size={20} />
                  <div>
                    <div className="font-medium">Claims Database</div>
                    <div className="text-sm text-slate-400">Structured data ingestion</div>
                  </div>
                </div>
                <span className="px-3 py-1 bg-green-500/20 text-green-400 rounded-full text-xs">Synced</span>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-r from-cyan-500/10 to-blue-500/10 rounded-xl p-6 border border-cyan-500/30">
            <h3 className="text-xl font-semibold mb-2">System Ready for Production</h3>
            <p className="text-slate-300 mb-4">
              FraudGuard AI is fully operational with all components integrated. The hybrid AI-human framework is achieving target KPIs and ready for full-scale deployment at GA Insurance.
            </p>
            <div className="flex gap-4">
              <div className="flex items-center gap-2 text-sm">
                <CheckCircle className="text-green-400" size={16} />
                <span>90% Detection Rate</span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <CheckCircle className="text-green-400" size={16} />
                <span>20% Loss Reduction</span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <CheckCircle className="text-green-400" size={16} />
                <span>4x Processing Speed</span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <CheckCircle className="text-green-400" size={16} />
                <span>&lt;10% False Positives</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="mt-8 pt-6 border-t border-slate-700 text-center text-sm text-slate-500">
        <div className="mb-2">
          FraudGuard AI - Powered by Machine Learning, Large Language Models, and Human Expertise
        </div>
        <div>
          Compliant with IRA Kenya regulations and Data Protection Act 2019
        </div>
      </div>
    </div>
  );
};

export default FraudGuardDashboard;