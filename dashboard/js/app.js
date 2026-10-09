// ==========================================================================
// AI-Based Dynamic Spectrum Sharing for Multi-MNO 6G Networks
// Executive Live Streaming JS Engine & Interactive Multi-Tab Navigation
// ==========================================================================

document.addEventListener("DOMContentLoaded", () => {
    // 1. Live Clock Engine
    const clockEl = document.getElementById("liveDateTime");
    function updateClock() {
        const now = new Date();
        const options = { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false };
        if (clockEl) {
            clockEl.textContent = now.toLocaleDateString('en-GB', options).replace(',', '');
        }
    }
    setInterval(updateClock, 1000);
    updateClock();

    // 2. Interactive Navigation Engine (Tab Switching)
    const sidebarMenuItems = document.querySelectorAll(".sidebar-menu .menu-item");
    const pageViews = document.querySelectorAll(".page-view");
    const pageMainTitle = document.getElementById("pageMainTitle");
    const pageSubTitle = document.getElementById("pageSubTitle");

    const pageTitlesMap = {
        "view-dashboard": { title: "AI-Based Dynamic Spectrum Sharing for Multi-MNO 6G Networks", subtitle: "Smarter Spectrum | Higher Efficiency | Connected Future" },
        "view-spectrum": { title: "Spectrum Overview & 16-Channel Sensing Map", subtitle: "Variational Autoencoder (VAE) + GMM Primary User Protection" },
        "view-mno": { title: "MNO Management & Dynamic Slicing", subtitle: "Airtel 6G, Jio 6G, Vi 6G, BSNL 6G Performance & SLA Compliance" },
        "view-ai": { title: "Deep Reinforcement Learning (DRL) Engine", subtitle: "Constraint Optimization (C1-C4) & PyTorch VAE-GMM Model Architecture" },
        "view-requests": { title: "Real-Time Bandwidth Requests & Event Logs", subtitle: "Live Stream of Spectrum Allocations and PU State Shifts" },
        "view-analytics": { title: "Spectrum Efficiency & Comparative Analytics Reports", subtitle: "Static Spectrum Slicing vs. AI Dynamic Sharing Benchmarks" },
        "view-settings": { title: "System Configuration & Database Settings", subtitle: "PostgreSQL Database Connection Status & Stream Controls" }
    };

    function switchTab(targetViewId) {
        pageViews.forEach(view => {
            if (view.id === targetViewId) {
                view.classList.add("active");
            } else {
                view.classList.remove("active");
            }
        });

        sidebarMenuItems.forEach(item => {
            if (item.getAttribute("data-target") === targetViewId) {
                item.classList.add("active");
            } else {
                item.classList.remove("active");
            }
        });

        if (pageTitlesMap[targetViewId]) {
            if (pageMainTitle) pageMainTitle.textContent = pageTitlesMap[targetViewId].title;
            if (pageSubTitle) pageSubTitle.textContent = pageTitlesMap[targetViewId].subtitle;
        }

        // Trigger chart resizes on active view
        window.dispatchEvent(new Event('resize'));
    }

    sidebarMenuItems.forEach(item => {
        item.addEventListener("click", (e) => {
            e.preventDefault();
            const target = item.getAttribute("data-target");
            if (target) switchTab(target);
        });
    });

    document.querySelectorAll(".switch-tab-link").forEach(link => {
        link.addEventListener("click", (e) => {
            e.preventDefault();
            const target = link.getAttribute("data-target");
            if (target) switchTab(target);
        });
    });

    // 3. Modal Controls (Admin Profile & Notifications)
    const adminModal = document.getElementById("adminModal");
    const btnAdminProfile = document.getElementById("btnAdminProfile");
    const closeAdminModal = document.getElementById("closeAdminModal");
    const btnNotifications = document.getElementById("btnNotifications");

    if (btnAdminProfile && adminModal) {
        btnAdminProfile.addEventListener("click", () => {
            adminModal.classList.add("active");
        });
    }

    if (btnNotifications && adminModal) {
        btnNotifications.addEventListener("click", () => {
            adminModal.classList.add("active");
        });
    }

    if (closeAdminModal && adminModal) {
        closeAdminModal.addEventListener("click", () => {
            adminModal.classList.remove("active");
        });
    }

    window.addEventListener("click", (e) => {
        if (e.target === adminModal) {
            adminModal.classList.remove("active");
        }
    });

    // 4. Main Executive Chart Instances
    let utilizationTimeChart = null;
    let spectrumDistributionChart = null;
    let bssComparisonChart = null;
    let requestAllocationChart = null;

    // Sub-page Charts
    let snrBarChart = null;
    let pbusyLineChart = null;
    let mnoHistoryChart = null;
    let aiRewardChart = null;
    let analyticsCompareChart = null;
    let fairnessTrendChart = null;

    // Initialize Dashboard Chart 1: Utilization Over Time
    const ctxUtil = document.getElementById("utilizationTimeChart")?.getContext("2d");
    if (ctxUtil) {
        utilizationTimeChart = new Chart(ctxUtil, {
            type: "line",
            data: {
                labels: ["08:00", "10:00", "12:00", "14:00", "16:00", "18:00", "20:00"],
                datasets: [
                    { label: "Airtel 6G", data: [22, 44, 38, 32, 46, 54, 39], borderColor: "#FF4D4D", fill: false, tension: 0.4, borderWidth: 2 },
                    { label: "Jio 6G", data: [34, 62, 55, 61, 74, 79, 64], borderColor: "#3B82F6", fill: false, tension: 0.4, borderWidth: 2 },
                    { label: "Vi 6G", data: [24, 38, 29, 26, 41, 44, 31], borderColor: "#F59E0B", fill: false, tension: 0.4, borderWidth: 2 },
                    { label: "BSNL 6G", data: [16, 29, 24, 20, 36, 37, 26], borderColor: "#A855F7", fill: false, tension: 0.4, borderWidth: 2 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: true, position: "top", labels: { color: "#94A3B8", font: { size: 10 } } } },
                scales: {
                    x: { ticks: { color: "#64748B", font: { size: 10 } }, grid: { color: "rgba(255, 255, 255, 0.05)" } },
                    y: { ticks: { color: "#64748B", font: { size: 10 } }, grid: { color: "rgba(255, 255, 255, 0.05)" }, min: 0, max: 100 }
                }
            }
        });
    }

    // Initialize Chart 2: Spectrum Distribution Doughnut
    const ctxDist = document.getElementById("spectrumDistributionChart")?.getContext("2d");
    if (ctxDist) {
        spectrumDistributionChart = new Chart(ctxDist, {
            type: "doughnut",
            data: {
                labels: ["Airtel 6G", "Jio 6G", "Vi 6G", "BSNL 6G"],
                datasets: [{
                    data: [245, 260, 238, 257],
                    backgroundColor: ["#FF4D4D", "#3B82F6", "#F59E0B", "#A855F7"],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: "70%",
                plugins: { legend: { position: "right", labels: { color: "#94A3B8", font: { size: 10 } } } }
            }
        });
    }

    // Initialize Chart 3: BSS Comparison Bar Chart
    const ctxBss = document.getElementById("bssComparisonChart")?.getContext("2d");
    if (ctxBss) {
        bssComparisonChart = new Chart(ctxBss, {
            type: "bar",
            data: {
                labels: ["Airtel 6G", "Jio 6G", "Vi 6G", "BSNL 6G"],
                datasets: [{
                    label: "BSS Score",
                    data: [0.82, 0.91, 0.76, 0.79],
                    backgroundColor: ["#FF4D4D", "#3B82F6", "#F59E0B", "#A855F7"],
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { ticks: { color: "#64748B", font: { size: 10 } }, grid: { display: false } },
                    y: { ticks: { color: "#64748B", font: { size: 10 } }, grid: { color: "rgba(255, 255, 255, 0.05)" }, min: 0, max: 1.0 }
                }
            }
        });
    }

    // Initialize Chart 4: Request vs Allocation
    const ctxReq = document.getElementById("requestAllocationChart")?.getContext("2d");
    if (ctxReq) {
        requestAllocationChart = new Chart(ctxReq, {
            type: "line",
            data: {
                labels: ["08:00", "10:00", "12:00", "14:00", "16:00", "18:00", "20:00"],
                datasets: [
                    { label: "Requests", data: [42, 65, 58, 48, 70, 76, 52], borderColor: "#3B82F6", tension: 0.4, borderWidth: 2 },
                    { label: "Allocated", data: [38, 56, 50, 42, 60, 65, 48], borderColor: "#10B981", borderDash: [4, 4], tension: 0.4, borderWidth: 2 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: true, position: "top", labels: { color: "#94A3B8", font: { size: 9 } } } },
                scales: {
                    x: { ticks: { color: "#64748B", font: { size: 9 } }, grid: { display: false } },
                    y: { ticks: { color: "#64748B", font: { size: 9 } }, grid: { color: "rgba(255, 255, 255, 0.05)" } }
                }
            }
        });
    }

    // Initialize Sub-Page Spectrum SNR Bar Chart
    const ctxSnr = document.getElementById("snrBarChart")?.getContext("2d");
    if (ctxSnr) {
        snrBarChart = new Chart(ctxSnr, {
            type: "bar",
            data: {
                labels: Array.from({length: 16}, (_, i) => `Ch ${i}`),
                datasets: [{
                    label: "SNR (dB)",
                    data: [32, 2.1, 28, 35, 1.8, 24, 31, 29, 2.5, 33, 27, 1.9, 30, 26, 34, 28],
                    backgroundColor: "#3B82F6",
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { ticks: { color: "#64748B", font: { size: 9 } }, grid: { display: false } },
                    y: { ticks: { color: "#64748B", font: { size: 9 } }, grid: { color: "rgba(255, 255, 255, 0.05)" } }
                }
            }
        });
    }

    // Initialize Sub-Page Spectrum P_busy Line Chart
    const ctxPbusy = document.getElementById("pbusyLineChart")?.getContext("2d");
    if (ctxPbusy) {
        pbusyLineChart = new Chart(ctxPbusy, {
            type: "line",
            data: {
                labels: Array.from({length: 16}, (_, i) => `Ch ${i}`),
                datasets: [{
                    label: "P_busy Probability",
                    data: [0.12, 0.95, 0.08, 0.14, 0.92, 0.18, 0.11, 0.09, 0.88, 0.10, 0.15, 0.94, 0.13, 0.16, 0.07, 0.12],
                    borderColor: "#EF4444",
                    backgroundColor: "rgba(239, 68, 68, 0.1)",
                    fill: true,
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { ticks: { color: "#64748B", font: { size: 9 } }, grid: { display: false } },
                    y: { ticks: { color: "#64748B", font: { size: 9 } }, grid: { color: "rgba(255, 255, 255, 0.05)" }, min: 0, max: 1.0 }
                }
            }
        });
    }

    // Initialize Sub-Page MNO History Chart
    const ctxMnoHist = document.getElementById("mnoHistoryChart")?.getContext("2d");
    if (ctxMnoHist) {
        mnoHistoryChart = new Chart(ctxMnoHist, {
            type: "line",
            data: {
                labels: ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"],
                datasets: [
                    { label: "Airtel 6G", data: [180, 140, 320, 480, 513, 390], borderColor: "#FF4D4D", tension: 0.4 },
                    { label: "Jio 6G", data: [240, 190, 450, 680, 728, 560], borderColor: "#3B82F6", tension: 0.4 },
                    { label: "Vi 6G", data: [130, 110, 210, 310, 327, 240], borderColor: "#F59E0B", tension: 0.4 },
                    { label: "BSNL 6G", data: [110, 95, 180, 270, 293, 210], borderColor: "#A855F7", tension: 0.4 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: "top", labels: { color: "#94A3B8", font: { size: 10 } } } },
                scales: {
                    x: { ticks: { color: "#64748B", font: { size: 10 } }, grid: { color: "rgba(255, 255, 255, 0.05)" } },
                    y: { ticks: { color: "#64748B", font: { size: 10 } }, grid: { color: "rgba(255, 255, 255, 0.05)" } }
                }
            }
        });
    }

    // Initialize Sub-Page AI DRL Reward Chart
    const ctxReward = document.getElementById("aiRewardChart")?.getContext("2d");
    if (ctxReward) {
        aiRewardChart = new Chart(ctxReward, {
            type: "line",
            data: {
                labels: Array.from({length: 20}, (_, i) => `Ep ${i*10}`),
                datasets: [{
                    label: "DRL Policy Cumulative Reward",
                    data: [-45, -30, -15, 10, 25, 45, 65, 80, 92, 105, 118, 125, 132, 138, 142, 145, 148, 150, 151, 152],
                    borderColor: "#10B981",
                    backgroundColor: "rgba(16, 185, 129, 0.1)",
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { ticks: { color: "#64748B", font: { size: 10 } }, grid: { display: false } },
                    y: { ticks: { color: "#64748B", font: { size: 10 } }, grid: { color: "rgba(255, 255, 255, 0.05)" } }
                }
            }
        });
    }

    // Initialize Sub-Page Analytics Compare Chart
    const ctxAnalytics = document.getElementById("analyticsCompareChart")?.getContext("2d");
    if (ctxAnalytics) {
        analyticsCompareChart = new Chart(ctxAnalytics, {
            type: "bar",
            data: {
                labels: ["Spectrum Utilization %", "Demand Satisfaction %", "Fairness Index (J*100)"],
                datasets: [
                    { label: "Static Slicing", data: [45.2, 58.4, 62.0], backgroundColor: "#EF4444", borderRadius: 4 },
                    { label: "AI Dynamic Sharing", data: [78.2, 94.6, 96.5], backgroundColor: "#10B981", borderRadius: 4 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: "top", labels: { color: "#94A3B8", font: { size: 10 } } } },
                scales: {
                    x: { ticks: { color: "#64748B", font: { size: 10 } }, grid: { display: false } },
                    y: { ticks: { color: "#64748B", font: { size: 10 } }, grid: { color: "rgba(255, 255, 255, 0.05)" }, min: 0, max: 100 }
                }
            }
        });
    }

    // Initialize Sub-Page Fairness Trend Chart
    const ctxFairness = document.getElementById("fairnessTrendChart")?.getContext("2d");
    if (ctxFairness) {
        fairnessTrendChart = new Chart(ctxFairness, {
            type: "line",
            data: {
                labels: ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"],
                datasets: [{
                    label: "Jain's Fairness Index (J)",
                    data: [0.94, 0.95, 0.97, 0.96, 0.98, 0.97],
                    borderColor: "#8B5CF6",
                    tension: 0.3,
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { ticks: { color: "#64748B", font: { size: 10 } }, grid: { display: false } },
                    y: { ticks: { color: "#64748B", font: { size: 10 } }, grid: { color: "rgba(255, 255, 255, 0.05)" }, min: 0.8, max: 1.0 }
                }
            }
        });
    }

    // 5. 16-Channel Heatmap Renderer for Spectrum Overview & Main Dashboard (Direct PostgreSQL Data)
    async function renderFullChannelGrid() {
        const gridEl = document.getElementById("fullChannelGrid");
        const mainGridEl = document.getElementById("mainDashboardChannelGrid");

        try {
            const res = await fetch("/api/spectrum_channels");
            if (!res.ok) return;
            const data = await res.json();

            let html = "";
            const snrValues = [];
            const pbusyValues = [];

            data.channels.forEach(ch => {
                const isPU = ch.pu_state === 1;
                const cardClass = isPU ? "pu-busy-card" : "free-allocated-card";
                snrValues.push(ch.snr_est_db);
                pbusyValues.push(ch.p_busy);

                html += `
                    <div class="ch-card ${cardClass}">
                        <div class="ch-title">Channel ${ch.channel_id < 10 ? '0' + ch.channel_id : ch.channel_id}</div>
                        <div class="ch-stat">${ch.status}</div>
                        <div class="ch-stat">P_busy: <strong>${ch.p_busy}</strong></div>
                        <div class="ch-stat">SNR: <strong>${ch.snr_est_db} dB</strong></div>
                        <div class="ch-stat">Noise: <strong>${ch.noise_est_dbm} dBm</strong></div>
                    </div>
                `;
            });

            if (gridEl) gridEl.innerHTML = html;
            if (mainGridEl) mainGridEl.innerHTML = html;

            if (snrBarChart && snrValues.length > 0) {
                snrBarChart.data.datasets[0].data = snrValues;
                snrBarChart.update("none");
            }
            if (pbusyLineChart && pbusyValues.length > 0) {
                pbusyLineChart.data.datasets[0].data = pbusyValues;
                pbusyLineChart.update("none");
            }
        } catch (err) {
            console.warn("Failed to fetch spectrum_channels:", err);
        }
    }


    // 6. Full Events Table Renderer
    function renderFullEventsTable(auditEvents) {
        const bodyEl = document.getElementById("fullEventsTableBody");
        if (!bodyEl) return;

        if (auditEvents && auditEvents.length > 0) {
            bodyEl.innerHTML = auditEvents.map(ev => `
                <tr>
                    <td><strong>${ev.id}</strong></td>
                    <td>${ev.time}</td>
                    <td><strong>${ev.mno}</strong></td>
                    <td>${ev.detail}</td>
                    <td>${ev.requested}</td>
                    <td>${ev.allocated}</td>
                    <td><span class="status-pill-table">${ev.status}</span></td>
                </tr>
            `).join("");
        }
    }

    // 7. Real-Time Data Sync from Flask API & PostgreSQL
    async function fetchDashboardData() {
        try {
            const res = await fetch("/api/dashboard_data");
            if (!res.ok) return;
            const data = await res.json();

            // Update KPIs
            if (document.getElementById("kpiTotalSpectrum")) document.getElementById("kpiTotalSpectrum").textContent = data.kpis.total_spectrum;
            if (document.getElementById("kpiUsedSpectrum")) document.getElementById("kpiUsedSpectrum").innerHTML = `${data.kpis.used_spectrum}`;
            if (document.getElementById("usedProgress")) document.getElementById("usedProgress").style.width = `${data.kpis.utilization_pct}%`;
            if (document.getElementById("kpiUsedPct")) document.getElementById("kpiUsedPct").textContent = `${data.kpis.utilization_pct}%`;
            if (document.getElementById("kpiPendingRequests")) document.getElementById("kpiPendingRequests").textContent = data.kpis.pending_requests;
            if (document.getElementById("kpiAvgBss")) document.getElementById("kpiAvgBss").textContent = data.kpis.avg_bss;

            // Render Active Pending Bandwidth Requests Table
            const pendingTableBody = document.getElementById("pendingRequestsTableBody");
            const pendingQueueCountHeader = document.getElementById("pendingQueueCountHeader");
            if (pendingQueueCountHeader) pendingQueueCountHeader.textContent = data.kpis.pending_requests;

            if (pendingTableBody && data.pending_requests_list) {
                if (data.pending_requests_list.length === 0) {
                    pendingTableBody.innerHTML = `
                        <tr>
                            <td colspan="7" style="text-align: center; color: #10B981; padding: 16px; font-size: 11px;">
                                🟢 <strong>Zero Spectrum Shortfall</strong> — All 4 MNO traffic demands are currently 100% satisfied by AI Dynamic Spectrum Sharing.
                            </td>
                        </tr>
                    `;
                } else {
                    pendingTableBody.innerHTML = data.pending_requests_list.map(req => `
                        <tr>
                            <td><strong>${req.id}</strong></td>
                            <td><strong>${req.mno}</strong></td>
                            <td>${req.requested_bw}</td>
                            <td>${req.allocated_bw}</td>
                            <td><strong style="color: #EF4444;">${req.shortfall}</strong></td>
                            <td><span style="font-size: 10px; font-weight: 600;">${req.priority}</span></td>
                            <td><span class="status-pill-table" style="color: #F59E0B;">🔸 ${req.status}</span></td>
                        </tr>
                    `).join("");
                }
            }
            if (data.mno_detailed) {
                const d = data.mno_detailed;
                if (d["Airtel 6G"]) {
                    if (document.getElementById("mnoAirtelUsers")) document.getElementById("mnoAirtelUsers").textContent = d["Airtel 6G"].users;
                    if (document.getElementById("mnoAirtelBw")) document.getElementById("mnoAirtelBw").textContent = d["Airtel 6G"].allocated;
                    if (document.getElementById("mnoAirtelDemand")) document.getElementById("mnoAirtelDemand").textContent = d["Airtel 6G"].demand;
                    if (document.getElementById("mnoAirtelBss")) document.getElementById("mnoAirtelBss").textContent = d["Airtel 6G"].bss;
                    if (document.getElementById("mnoAirtelSla")) document.getElementById("mnoAirtelSla").textContent = d["Airtel 6G"].sla;
                }
                if (d["Jio 6G"]) {
                    if (document.getElementById("mnoJioUsers")) document.getElementById("mnoJioUsers").textContent = d["Jio 6G"].users;
                    if (document.getElementById("mnoJioBw")) document.getElementById("mnoJioBw").textContent = d["Jio 6G"].allocated;
                    if (document.getElementById("mnoJioDemand")) document.getElementById("mnoJioDemand").textContent = d["Jio 6G"].demand;
                    if (document.getElementById("mnoJioBss")) document.getElementById("mnoJioBss").textContent = d["Jio 6G"].bss;
                    if (document.getElementById("mnoJioSla")) document.getElementById("mnoJioSla").textContent = d["Jio 6G"].sla;
                }
                if (d["Vi 6G"]) {
                    if (document.getElementById("mnoViUsers")) document.getElementById("mnoViUsers").textContent = d["Vi 6G"].users;
                    if (document.getElementById("mnoViBw")) document.getElementById("mnoViBw").textContent = d["Vi 6G"].allocated;
                    if (document.getElementById("mnoViDemand")) document.getElementById("mnoViDemand").textContent = d["Vi 6G"].demand;
                    if (document.getElementById("mnoViBss")) document.getElementById("mnoViBss").textContent = d["Vi 6G"].bss;
                    if (document.getElementById("mnoViSla")) document.getElementById("mnoViSla").textContent = d["Vi 6G"].sla;
                }
                if (d["BSNL 6G"]) {
                    if (document.getElementById("mnoBsnlUsers")) document.getElementById("mnoBsnlUsers").textContent = d["BSNL 6G"].users;
                    if (document.getElementById("mnoBsnlBw")) document.getElementById("mnoBsnlBw").textContent = d["BSNL 6G"].allocated;
                    if (document.getElementById("mnoBsnlDemand")) document.getElementById("mnoBsnlDemand").textContent = d["BSNL 6G"].demand;
                    if (document.getElementById("mnoBsnlBss")) document.getElementById("mnoBsnlBss").textContent = d["BSNL 6G"].bss;
                    if (document.getElementById("mnoBsnlSla")) document.getElementById("mnoBsnlSla").textContent = d["BSNL 6G"].sla;
                }
            }

            // Update AI Agent (DRL) Sub-Page Dynamic Metrics & Chart
            if (data.ai_engine_metrics) {
                if (document.getElementById("aiMseLoss")) document.getElementById("aiMseLoss").textContent = data.ai_engine_metrics.mse_loss;
                if (document.getElementById("aiPuInterference")) document.getElementById("aiPuInterference").textContent = data.ai_engine_metrics.pu_interference;
                if (document.getElementById("aiFairnessIndex")) document.getElementById("aiFairnessIndex").textContent = data.ai_engine_metrics.fairness_index;
                if (document.getElementById("aiStepBadge")) document.getElementById("aiStepBadge").textContent = `🟢 Step #${data.step_idx} Active`;

                if (aiRewardChart && data.ai_engine_metrics.reward_series) {
                    aiRewardChart.data.labels = data.ai_engine_metrics.reward_labels;
                    aiRewardChart.data.datasets[0].data = data.ai_engine_metrics.reward_series;
                    aiRewardChart.update("none");
                }
            }

            // Update Analytics & Reports Tab Charts
            if (data.analytics_data) {
                if (analyticsCompareChart && data.analytics_data.static_vs_dynamic) {
                    analyticsCompareChart.data.datasets[1].data = data.analytics_data.static_vs_dynamic.dynamic;
                    analyticsCompareChart.update("none");
                }
                if (fairnessTrendChart && data.analytics_data.fairness_trend) {
                    fairnessTrendChart.data.labels = data.analytics_data.fairness_trend.labels;
                    fairnessTrendChart.data.datasets[0].data = data.analytics_data.fairness_trend.series;
                    fairnessTrendChart.update("none");
                }
            }

            // Update Utilization Over Time Chart
            if (utilizationTimeChart && data.utilization_over_time) {
                utilizationTimeChart.data.labels = data.utilization_over_time.labels;
                if (data.utilization_over_time.series["Airtel 6G"]) utilizationTimeChart.data.datasets[0].data = data.utilization_over_time.series["Airtel 6G"];
                if (data.utilization_over_time.series["Jio 6G"]) utilizationTimeChart.data.datasets[1].data = data.utilization_over_time.series["Jio 6G"];
                if (data.utilization_over_time.series["Vi 6G"]) utilizationTimeChart.data.datasets[2].data = data.utilization_over_time.series["Vi 6G"];
                if (data.utilization_over_time.series["BSNL 6G"]) utilizationTimeChart.data.datasets[3].data = data.utilization_over_time.series["BSNL 6G"];
                utilizationTimeChart.update("none");
            }

            // Update MNO History Chart on MNO Management Page
            if (mnoHistoryChart && data.utilization_over_time) {
                mnoHistoryChart.data.labels = data.utilization_over_time.labels;
                if (data.utilization_over_time.series["Airtel 6G"]) mnoHistoryChart.data.datasets[0].data = data.utilization_over_time.series["Airtel 6G"].map(x => Math.round(x * 6.0));
                if (data.utilization_over_time.series["Jio 6G"]) mnoHistoryChart.data.datasets[1].data = data.utilization_over_time.series["Jio 6G"].map(x => Math.round(x * 8.5));
                if (data.utilization_over_time.series["Vi 6G"]) mnoHistoryChart.data.datasets[2].data = data.utilization_over_time.series["Vi 6G"].map(x => Math.round(x * 4.5));
                if (data.utilization_over_time.series["BSNL 6G"]) mnoHistoryChart.data.datasets[3].data = data.utilization_over_time.series["BSNL 6G"].map(x => Math.round(x * 3.8));
                mnoHistoryChart.update("none");
            }

            // Update Spectrum Distribution Chart
            if (spectrumDistributionChart && data.spectrum_distribution) {
                spectrumDistributionChart.data.datasets[0].data = data.spectrum_distribution.map(d => d.mhz);
                spectrumDistributionChart.update("none");
            }

            // Update BSS Comparison Chart
            if (bssComparisonChart && data.bss_comparison) {
                bssComparisonChart.data.datasets[0].data = data.bss_comparison.map(d => d.score);
                bssComparisonChart.update("none");
            }

            // Update Requests vs Allocation Chart
            if (requestAllocationChart && data.requests_vs_allocation) {
                requestAllocationChart.data.datasets[0].data = data.requests_vs_allocation.requests;
                requestAllocationChart.data.datasets[1].data = data.requests_vs_allocation.allocated;
                requestAllocationChart.update("none");
            }

            // Update Live Events Feed
            const eventsListEl = document.getElementById("eventsFeedList");
            if (eventsListEl && data.live_events) {
                eventsListEl.innerHTML = data.live_events.map(ev => `
                    <div class="event-item ${ev.type || 'info'}">
                        <span class="event-icon">${ev.icon}</span>
                        <div class="event-content">
                            <span class="event-title">${ev.title}</span>
                            <span class="event-detail">${ev.detail}</span>
                        </div>
                        <span class="event-time">${ev.time}</span>
                    </div>
                `).join("");
            }

            // Update MNO Performance Table
            const mnoTableBody = document.getElementById("mnoPerformanceBody");
            if (mnoTableBody && data.mno_performance) {
                mnoTableBody.innerHTML = data.mno_performance.map(row => `
                    <tr>
                        <td><span class="mno-dot" style="background-color: ${row.color};"></span> <strong>${row.mno}</strong></td>
                        <td>${row.used_spectrum}</td>
                        <td>${row.total_spectrum}</td>
                        <td>${row.utilization}</td>
                        <td><strong>${row.bss}</strong></td>
                        <td><span class="status-pill-table">🟢 ${row.status}</span></td>
                    </tr>
                `).join("");
            }

            // Update Recent AI Decisions Table
            const aiDecisionsBody = document.getElementById("aiDecisionsBody");
            if (aiDecisionsBody && data.recent_ai_decisions) {
                aiDecisionsBody.innerHTML = data.recent_ai_decisions.map(d => `
                    <tr>
                        <td>${d.time}</td>
                        <td>${d.event}</td>
                        <td>${d.action}</td>
                    </tr>
                `).join("");
            }

            // Render 16 Channel Grid & Full Events Audit Trail Table
            renderFullChannelGrid();
            renderFullEventsTable(data.audit_events_list);

        } catch (err) {
            console.warn("Failed to update dashboard from API:", err);
        }
    }

    // Poll every 3 seconds for live streaming updates
    setInterval(fetchDashboardData, 3000);
    fetchDashboardData();
});

