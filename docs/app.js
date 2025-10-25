// Application Data
const portfolioData = {
  "portfolio_summary": {
    "total_loans": 1000,
    "total_exposure": 19707696.60,
    "avg_default_rate": 0.2061,
    "total_expected_loss": 1805517.78,
    "grade_distribution": {"A": 207, "B": 333, "C": 249, "D": 199, "E": 12}
  },
  "sample_loans": [
    {"loan_id": "LC000001", "amount": 15000, "grade": "B", "default_prob": 0.18, "purpose": "debt_consolidation"},
    {"loan_id": "LC000002", "amount": 22000, "grade": "C", "default_prob": 0.25, "purpose": "credit_card"},
    {"loan_id": "LC000003", "amount": 8000, "grade": "A", "default_prob": 0.08, "purpose": "home_improvement"},
    {"loan_id": "LC000004", "amount": 35000, "grade": "D", "default_prob": 0.42, "purpose": "debt_consolidation"},
    {"loan_id": "LC000005", "amount": 12000, "grade": "B", "default_prob": 0.16, "purpose": "other"}
  ],
  "performance_metrics": {
    "auc_lr": 0.7406,
    "auc_rf": 0.6997,
    "gini_lr": 0.4812,
    "gini_rf": 0.3994
  },
  "feature_importance": [
    {"feature": "Interest Rate", "importance": 0.113},
    {"feature": "DTI Ratio", "importance": 0.103},
    {"feature": "Revolving Balance", "importance": 0.073},
    {"feature": "Utilization Ratio", "importance": 0.069},
    {"feature": "Loan Amount", "importance": 0.069},
    {"feature": "Annual Income", "importance": 0.069}
  ],
  "time_series": [
    {"date": "2023-01", "default_rate": 0.18, "new_loans": 120, "portfolio_value": 65000000},
    {"date": "2023-02", "default_rate": 0.19, "new_loans": 135, "portfolio_value": 67000000},
    {"date": "2023-03", "default_rate": 0.21, "new_loans": 110, "portfolio_value": 69000000},
    {"date": "2024-01", "default_rate": 0.20, "new_loans": 125, "portfolio_value": 72000000},
    {"date": "2024-12", "default_rate": 0.22, "new_loans": 140, "portfolio_value": 75000000}
  ]
};

// Chart color palette
const chartColors = ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F', '#DB4545', '#D2BA4C', '#964325', '#944454', '#13343B'];

// Global chart instances
let charts = {};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeNavigation();
    initializeSliders();
    initializeScorer();
    initializeCharts();
    populateRecentLoansTable();
});

// Navigation functionality
function initializeNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const tabContents = document.querySelectorAll('.tab-content');

    navItems.forEach(item => {
        item.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Remove active class from all nav items and tabs
            navItems.forEach(nav => nav.classList.remove('active'));
            tabContents.forEach(tab => tab.classList.remove('active'));
            
            // Add active class to clicked nav item
            this.classList.add('active');
            
            // Show corresponding tab
            const tabId = this.getAttribute('data-tab');
            const targetTab = document.getElementById(tabId);
            if (targetTab) {
                targetTab.classList.add('active');
                
                // Initialize charts for the active tab
                setTimeout(() => {
                    if (tabId === 'analytics') {
                        initializeAnalyticsCharts();
                    } else if (tabId === 'monitoring') {
                        initializeMonitoringCharts();
                    } else if (tabId === 'performance') {
                        initializePerformanceCharts();
                    }
                }, 100);
            }
        });
    });
}

// Initialize range sliders
function initializeSliders() {
    const loanAmountSlider = document.getElementById('loanAmount');
    const loanAmountValue = document.getElementById('loanAmountValue');
    const dtiRatioSlider = document.getElementById('dtiRatio');
    const dtiRatioValue = document.getElementById('dtiRatioValue');
    const interestRateSlider = document.getElementById('interestRate');
    const interestRateValue = document.getElementById('interestRateValue');

    if (loanAmountSlider && loanAmountValue) {
        loanAmountSlider.addEventListener('input', function() {
            loanAmountValue.textContent = `$${parseInt(this.value).toLocaleString()}`;
        });
    }

    if (dtiRatioSlider && dtiRatioValue) {
        dtiRatioSlider.addEventListener('input', function() {
            dtiRatioValue.textContent = `${this.value}%`;
        });
    }

    if (interestRateSlider && interestRateValue) {
        interestRateSlider.addEventListener('input', function() {
            interestRateValue.textContent = `${parseFloat(this.value).toFixed(1)}%`;
        });
    }
}

// Initialize loan scorer functionality
function initializeScorer() {
    const calculateButton = document.getElementById('calculateRisk');
    if (calculateButton) {
        calculateButton.addEventListener('click', calculateRiskScore);
    }
}

// Calculate risk score based on form inputs
function calculateRiskScore() {
    const loanAmount = parseInt(document.getElementById('loanAmount').value);
    const annualIncome = parseInt(document.getElementById('annualIncome').value);
    const dtiRatio = parseInt(document.getElementById('dtiRatio').value);
    const empLength = parseInt(document.getElementById('empLength').value);
    const homeOwnership = document.querySelector('input[name="homeOwnership"]:checked').value;
    const loanPurpose = document.getElementById('loanPurpose').value;
    const creditGrade = document.getElementById('creditGrade').value;
    const interestRate = parseFloat(document.getElementById('interestRate').value);

    // Simple risk scoring algorithm (mock implementation)
    let baseRisk = 0.15;
    
    // Grade-based risk adjustment
    const gradeRisk = {
        'A': 0.05, 'B': 0.15, 'C': 0.25, 'D': 0.35, 'E': 0.45, 'F': 0.55, 'G': 0.65
    };
    baseRisk = gradeRisk[creditGrade] || 0.25;
    
    // DTI adjustment
    baseRisk += (dtiRatio / 100) * 0.5;
    
    // Interest rate adjustment
    baseRisk += (interestRate - 10) * 0.02;
    
    // Employment length adjustment
    if (empLength < 2) baseRisk += 0.05;
    if (empLength >= 5) baseRisk -= 0.03;
    
    // Home ownership adjustment
    if (homeOwnership === 'OWN') baseRisk -= 0.02;
    if (homeOwnership === 'RENT') baseRisk += 0.01;
    
    // Loan amount vs income ratio
    const loanToIncomeRatio = loanAmount / annualIncome;
    if (loanToIncomeRatio > 0.5) baseRisk += 0.1;
    
    // Cap the risk between 0.01 and 0.95
    const defaultProb = Math.max(0.01, Math.min(0.95, baseRisk));
    
    // Calculate other metrics
    const creditScore = Math.round(850 - (defaultProb * 550));
    const expectedLoss = loanAmount * defaultProb;
    
    let riskGrade, recommendation;
    if (defaultProb < 0.1) {
        riskGrade = 'Low Risk';
        recommendation = 'APPROVE';
    } else if (defaultProb < 0.3) {
        riskGrade = 'Medium Risk';
        recommendation = 'APPROVE with conditions';
    } else {
        riskGrade = 'High Risk';
        recommendation = 'DECLINE';
    }

    // Update results display
    document.getElementById('defaultProb').textContent = `${(defaultProb * 100).toFixed(1)}%`;
    document.getElementById('creditScore').textContent = creditScore;
    document.getElementById('riskGrade').textContent = riskGrade;
    document.getElementById('expectedLoss').textContent = `$${expectedLoss.toLocaleString()}`;
    document.getElementById('recommendation').textContent = recommendation;

    // Show results section
    document.getElementById('resultsSection').classList.remove('hidden');
    document.getElementById('placeholder').classList.add('hidden');
}

// Initialize main dashboard charts
function initializeCharts() {
    initializeGradeChart();
    initializeHeatMapChart();
}

// Portfolio grade distribution chart
function initializeGradeChart() {
    const ctx = document.getElementById('gradeChart');
    if (!ctx) return;

    const gradeData = portfolioData.portfolio_summary.grade_distribution;
    
    charts.gradeChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: Object.keys(gradeData),
            datasets: [{
                data: Object.values(gradeData),
                backgroundColor: chartColors.slice(0, Object.keys(gradeData).length),
                borderWidth: 2,
                borderColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        usePointStyle: true
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = ((context.parsed / total) * 100).toFixed(1);
                            return `Grade ${context.label}: ${context.parsed} loans (${percentage}%)`;
                        }
                    }
                }
            }
        }
    });
}

// Risk heat map chart (simplified as bar chart)
function initializeHeatMapChart() {
    const ctx = document.getElementById('heatMapChart');
    if (!ctx) return;

    charts.heatMapChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Debt Consolidation', 'Credit Card', 'Home Improvement', 'Other', 'Major Purchase'],
            datasets: [{
                label: 'Average Risk Score',
                data: [0.23, 0.28, 0.18, 0.25, 0.21],
                backgroundColor: chartColors[0],
                borderColor: chartColors[0],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 0.5,
                    ticks: {
                        callback: function(value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Risk Score: ${(context.parsed.y * 100).toFixed(1)}%`;
                        }
                    }
                }
            }
        }
    });
}

// Initialize analytics charts
function initializeAnalyticsCharts() {
    if (!charts.portfolioGradeChart) {
        initializePortfolioGradeChart();
    }
    if (!charts.portfolioPurposeChart) {
        initializePortfolioPurposeChart();
    }
    if (!charts.riskDistributionChart) {
        initializeRiskDistributionChart();
    }
}

function initializePortfolioGradeChart() {
    const ctx = document.getElementById('portfolioGradeChart');
    if (!ctx) return;

    const gradeData = portfolioData.portfolio_summary.grade_distribution;
    
    charts.portfolioGradeChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: Object.keys(gradeData),
            datasets: [{
                data: Object.values(gradeData),
                backgroundColor: chartColors.slice(0, Object.keys(gradeData).length),
                borderWidth: 2,
                borderColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right'
                }
            }
        }
    });
}

function initializePortfolioPurposeChart() {
    const ctx = document.getElementById('portfolioPurposeChart');
    if (!ctx) return;

    charts.portfolioPurposeChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Debt Consolidation', 'Credit Card', 'Home Improvement', 'Other'],
            datasets: [{
                label: 'Number of Loans',
                data: [450, 250, 200, 100],
                backgroundColor: chartColors[1],
                borderColor: chartColors[1],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

function initializeRiskDistributionChart() {
    const ctx = document.getElementById('riskDistributionChart');
    if (!ctx) return;

    charts.riskDistributionChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50%+'],
            datasets: [{
                label: 'Number of Loans',
                data: [150, 300, 250, 200, 80, 20],
                backgroundColor: chartColors[2],
                borderColor: chartColors[2],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Distribution of Default Probability'
                }
            }
        }
    });
}

// Initialize monitoring charts
function initializeMonitoringCharts() {
    if (!charts.defaultRatesChart) {
        initializeDefaultRatesChart();
    }
    if (!charts.portfolioValueChart) {
        initializePortfolioValueChart();
    }
}

function initializeDefaultRatesChart() {
    const ctx = document.getElementById('defaultRatesChart');
    if (!ctx) return;

    const timeSeriesData = portfolioData.time_series;
    
    charts.defaultRatesChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: timeSeriesData.map(d => d.date),
            datasets: [{
                label: 'Default Rate',
                data: timeSeriesData.map(d => d.default_rate * 100),
                borderColor: chartColors[3],
                backgroundColor: chartColors[3] + '20',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Default Rate: ${context.parsed.y.toFixed(1)}%`;
                        }
                    }
                }
            }
        }
    });
}

function initializePortfolioValueChart() {
    const ctx = document.getElementById('portfolioValueChart');
    if (!ctx) return;

    const timeSeriesData = portfolioData.time_series;
    
    charts.portfolioValueChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: timeSeriesData.map(d => d.date),
            datasets: [{
                label: 'Portfolio Value',
                data: timeSeriesData.map(d => d.portfolio_value / 1000000),
                borderColor: chartColors[4],
                backgroundColor: chartColors[4] + '20',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return '$' + value + 'M';
                        }
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Portfolio Value: $${context.parsed.y.toFixed(1)}M`;
                        }
                    }
                }
            }
        }
    });
}

// Initialize performance charts
function initializePerformanceCharts() {
    if (!charts.featureImportanceChart) {
        initializeFeatureImportanceChart();
    }
}

function initializeFeatureImportanceChart() {
    const ctx = document.getElementById('featureImportanceChart');
    if (!ctx) return;

    const featureData = portfolioData.feature_importance;
    
    charts.featureImportanceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: featureData.map(d => d.feature),
            datasets: [{
                label: 'Importance',
                data: featureData.map(d => d.importance),
                backgroundColor: chartColors[5],
                borderColor: chartColors[5],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            scales: {
                x: {
                    beginAtZero: true,
                    max: 0.15,
                    ticks: {
                        callback: function(value) {
                            return (value * 100).toFixed(1) + '%';
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Importance: ${(context.parsed.x * 100).toFixed(1)}%`;
                        }
                    }
                }
            }
        }
    });
}

// Populate recent loans table
function populateRecentLoansTable() {
    const tableBody = document.getElementById('recentLoansTable');
    if (!tableBody) return;

    const sampleLoans = portfolioData.sample_loans;
    
    tableBody.innerHTML = sampleLoans.map(loan => `
        <tr>
            <td>${loan.loan_id}</td>
            <td>$${loan.amount.toLocaleString()}</td>
            <td><span class="status status--${getRiskClass(loan.grade)}">${loan.grade}</span></td>
            <td>${(loan.default_prob * 100).toFixed(1)}%</td>
            <td>${formatPurpose(loan.purpose)}</td>
        </tr>
    `).join('');
}

// Helper functions
function getRiskClass(grade) {
    if (grade === 'A' || grade === 'B') return 'low-risk';
    if (grade === 'C' || grade === 'D') return 'medium-risk';
    return 'high-risk';
}

function formatPurpose(purpose) {
    return purpose.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

// Filter functionality for analytics
document.addEventListener('change', function(e) {
    if (e.target.id === 'gradeFilter' || e.target.id === 'purposeFilter') {
        // In a real application, this would filter the data and update charts
        console.log('Filter applied:', e.target.value);
    }
});

// Resize charts on window resize
window.addEventListener('resize', function() {
    Object.values(charts).forEach(chart => {
        if (chart && typeof chart.resize === 'function') {
            chart.resize();
        }
    });
});