/**
 * AI-Based Dynamic Spectrum Sharing for Multi-MNO 6G Networks
 * Real-Time Dashboard Engine & Interactive Controller
 */

document.addEventListener("DOMContentLoaded", () => {
    // -------------------------------------------------------------------------
    // 1. Live Clock & Navigation Controller
    // -------------------------------------------------------------------------
    const clockEl = document.getElementById("liveDateTime");

    function updateLiveClock() {
        if (!clockEl) return;
        const now = new Date();
        const options = {
            day: "2-digit",
            month: "short",
            year: "numeric",
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
            hour12: false
        };
        clockEl.textContent = now.toLocaleDateString("en-GB", options).replace(",", "");
    }

    setInterval(updateLiveClock, 1000);
    updateLiveClock();

    const pageTitlesMap = {
        "view-dashboard": {
            title: "AI-Based Dynamic Spectrum Sharing for Multi-MNO 6G Networks",
            subtitle: "Smarter Spectrum | Higher Efficiency | Connected Future"
        },
        "view-spectrum": {
            title: "Spectrum Overview & 16-Channel Sensing Map",
            subtitle: "Variational Autoencoder (VAE) + GMM Primary User Protection"
        },
        "view-mno": {
            title: "MNO Management & Dynamic Slicing",
            subtitle: "Airtel 6G, Jio 6G, Vi 6G, BSNL 6G Performance & SLA Compliance"
        },
        "view-ai": {
            title: "Deep Reinforcement Learning (DRL) Engine",
            subtitle: "Constraint Optimization (C1-C4) & PyTorch VAE-GMM Model Architecture"
        },
        "view-requests": {
            title: "Real-Time Bandwidth Requests & Event Logs",
            subtitle: "Live Stream of Spectrum Allocations and PU State Shifts"
        },
        "view-analytics": {
            title: "Spectrum Efficiency & Comparative Analytics Reports",
            subtitle: "Static Spectrum Slicing vs. AI Dynamic Sharing Benchmarks"
        },
        "view-settings": {
            title: "System Configuration & Database Settings",
            subtitle: "PostgreSQL Database Connection Status & Stream Controls"
        }
    };

    const sidebarMenuItems = document.querySelectorAll(".sidebar-menu .menu-item");
    const pageViews = document.querySelectorAll(".page-view");
    const pageMainTitle = document.getElementById("pageMainTitle");
    const pageSubTitle = document.getElementById("pageSubTitle");

    function switchView(targetViewId) {
        pageViews.forEach(view => {
            view.classList.toggle("active", view.id === targetViewId);
        });

        sidebarMenuItems.forEach(item => {
            item.classList.toggle("active", item.getAttribute("data-target") === targetViewId);
        });

        const meta = pageTitlesMap[targetViewId];
        if (meta) {
            if (pageMainTitle) pageMainTitle.textContent = meta.title;
            if (pageSubTitle) pageSubTitle.textContent = meta.subtitle;
        }

        window.dispatchEvent(new Event("resize"));
    }

    sidebarMenuItems.forEach(item => {
        item.addEventListener("click", (e) => {
            e.preventDefault();
            const target = item.getAttribute("data-target");
            if (target) switchView(target);
        });
    });

    document.querySelectorAll(".switch-tab-link").forEach(link => {
        link.addEventListener("click", (e) => {
            e.preventDefault();
            const target = link.getAttribute("data-target");
            if (target) switchView(target);
        });
    });

    // -------------------------------------------------------------------------
    // 2. Modals (Admin Profile)
    // -------------------------------------------------------------------------
    const adminModal = document.getElementById("adminModal");
    const btnAdminProfile = document.getElementById("btnAdminProfile");
    const btnNotifications = document.getElementById("btnNotifications");
    const closeAdminModal = document.getElementById("closeAdminModal");

    const openModal = () => adminModal?.classList.add("active");
    const closeModal = () => adminModal?.classList.remove("active");

    btnAdminProfile?.addEventListener("click", openModal);
    btnNotifications?.addEventListener("click", openModal);
    closeAdminModal?.addEventListener("click", closeModal);

    window.addEventListener("click", (e) => {
        if (e.target === adminModal) closeModal();
    });

    // -------------------------------------------------------------------------
    // 3. Chart Factory Helpers & Instances
    // -------------------------------------------------------------------------
    const commonChartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: { color: "#94A3B8", font: { size: 10 } }
            }
        },
        scales: {
            x: {
                ticks: { color: "#64748B", font: { size: 9 } },
                grid: { color: "rgba(255, 255, 255, 0.05)" }
            },
            y: {
                ticks: { color: "#64748B", font: { size: 9 } },
                grid: { color: "rgba(255, 255, 255, 0.05)" }
            }
        }
    };

    function initLineChart(canvasId, labels, datasets, yMin = null, yMax = null, hideLegend = false) {
        const ctx = document.getElementById(canvasId)?.getContext("2d");
        if (!ctx) return null;

        const options = JSON.parse(JSON.stringify(commonChartOptions));
        if (hideLegend) options.plugins.legend.display = false;
        if (yMin !== null) options.scales.y.min = yMin;
        if (yMax !== null) options.scales.y.max = yMax;

        return new Chart(ctx, { type: "line", data: { labels, datasets }, options });
    }

    function initBarChart(canvasId, labels, datasets, yMax = null, hideLegend = true) {
        const ctx = document.getElementById(canvasId)?.getContext("2d");
        if (!ctx) return null;

        const options = JSON.parse(JSON.stringify(commonChartOptions));
        if (hideLegend) options.plugins.legend.display = false;
        if (yMax !== null) options.scales.y.max = yMax;

        return new Chart(ctx, { type: "bar", data: { labels, datasets }, options });
    }

    // Chart initializations
    const utilizationTimeChart = initLineChart("utilizationTimeChart",
        ["08:00", "10:00", "12:00", "14:00", "16:00", "18:00", "20:00"],
        [
            { label: "Airtel 6G", data: [22, 44, 38, 32, 46, 54, 39], borderColor: "#FF4D4D", tension: 0.4, borderWidth: 2 },
            { label: "Jio 6G", data: [34, 62, 55, 61, 74, 79, 64], borderColor: "#3B82F6", tension: 0.4, borderWidth: 2 },
            { label: "Vi 6G", data: [24, 38, 29, 26, 41, 44, 31], borderColor: "#F59E0B", tension: 0.4, borderWidth: 2 },
            { label: "BSNL 6G", data: [16, 29, 24, 20, 36, 37, 26], borderColor: "#A855F7", tension: 0.4, borderWidth: 2 }
        ], 0, 100
    );

    const spectrumDistributionChart = (() => {
        const ctx = document.getElementById("spectrumDistributionChart")?.getContext("2d");
        if (!ctx) return null;
        return new Chart(ctx, {
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
                plugins: {
                    legend: { position: "right", labels: { color: "#94A3B8", font: { size: 10 } } }
                }
            }
        });
    })();

    const bssComparisonChart = initBarChart("bssComparisonChart",
        ["Airtel 6G", "Jio 6G", "Vi 6G", "BSNL 6G"],
        [{
            label: "BSS Score",
            data: [0.82, 0.91, 0.76, 0.79],
            backgroundColor: ["#FF4D4D", "#3B82F6", "#F59E0B", "#A855F7"],
            borderRadius: 4
        }], 1.0
    );

    const requestAllocationChart = initLineChart("requestAllocationChart",
        ["08:00", "10:00", "12:00", "14:00", "16:00", "18:00", "20:00"],
        [
            { label: "Requests", data: [42, 65, 58, 48, 70, 76, 52], borderColor: "#3B82F6", tension: 0.4, borderWidth: 2 },
            { label: "Allocated", data: [38, 56, 50, 42, 60, 65, 48], borderColor: "#10B981", borderDash: [4, 4], tension: 0.4, borderWidth: 2 }
        ]
    );

    const snrBarChart = initBarChart("snrBarChart",
        Array.from({ length: 16 }, (_, i) => `Ch ${i}`),
        [{
            label: "SNR (dB)",
            data: [32, 2.1, 28, 35, 1.8, 24, 31, 29, 2.5, 33, 27, 1.9, 30, 26, 34, 28],
            backgroundColor: "#3B82F6",
            borderRadius: 4
        }]
    );

    const pbusyLineChart = initLineChart("pbusyLineChart",
        Array.from({ length: 16 }, (_, i) => `Ch ${i}`),
        [{
            label: "P_busy Probability",
            data: [0.12, 0.95, 0.08, 0.14, 0.92, 0.18, 0.11, 0.09, 0.88, 0.10, 0.15, 0.94, 0.13, 0.16, 0.07, 0.12],
            borderColor: "#EF4444",
            backgroundColor: "rgba(239, 68, 68, 0.1)",
            fill: true,
            tension: 0.3
        }], 0, 1.0, true
    );

    const mnoHistoryChart = initLineChart("mnoHistoryChart",
        ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"],
        [
            { label: "Airtel 6G", data: [180, 140, 320, 480, 513, 390], borderColor: "#FF4D4D", tension: 0.4 },
            { label: "Jio 6G", data: [240, 190, 450, 680, 728, 560], borderColor: "#3B82F6", tension: 0.4 },
            { label: "Vi 6G", data: [130, 110, 210, 310, 327, 240], borderColor: "#F59E0B", tension: 0.4 },
            { label: "BSNL 6G", data: [110, 95, 180, 270, 293, 210], borderColor: "#A855F7", tension: 0.4 }
        ]
    );

    const aiRewardChart = initLineChart("aiRewardChart",
        Array.from({ length: 11 }, (_, i) => `Step #${i}`),
        [{
            label: "DRL Cumulative Reward",
            data: [115, 120, 124, 129, 133, 137, 141, 144, 148, 150, 153],
            borderColor: "#10B981",
            backgroundColor: "rgba(16, 185, 129, 0.1)",
            fill: true,
            tension: 0.4
        }], null, null, true
    );

    const analyticsCompareChart = initBarChart("analyticsCompareChart",
        ["Spectrum Utilization %", "Demand Satisfaction %", "Fairness Index (J*100)"],
        [
            { label: "Static Slicing", data: [45.2, 58.4, 62.0], backgroundColor: "#EF4444", borderRadius: 4 },
            { label: "AI Dynamic Sharing", data: [78.2, 94.6, 96.5], backgroundColor: "#10B981", borderRadius: 4 }
        ], 100, false
    );

    const fairnessTrendChart = initLineChart("fairnessTrendChart",
        ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"],
        [{
            label: "Jain's Fairness Index (J)",
            data: [0.94, 0.95, 0.97, 0.96, 0.98, 0.97],
            borderColor: "#8B5CF6",
            tension: 0.3,
            borderWidth: 2
        }], 0.8, 1.0, true
    );

    // -------------------------------------------------------------------------
    // 4. DOM Update Routines & Renderers
    // -------------------------------------------------------------------------
    function setText(id, text) {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    }

    function renderChannelGrid(channels) {
        if (!Array.isArray(channels) || channels.length === 0) return;

        const mainGrid = document.getElementById("mainDashboardChannelGrid");
        const fullGrid = document.getElementById("fullChannelGrid");

        const snrValues = [];
        const pbusyValues = [];

        const html = channels.map(ch => {
            const isPU = ch.pu_state === 1;
            const cardClass = isPU ? "pu-busy-card" : "free-allocated-card";
            snrValues.push(ch.snr_est_db);
            pbusyValues.push(ch.p_busy);

            return `
                <div class="ch-card ${cardClass}">
                    <div class="ch-title">Channel ${String(ch.channel_id).padStart(2, "0")}</div>
                    <div class="ch-stat">${ch.status}</div>
                    <div class="ch-stat">P_busy: <strong>${ch.p_busy}</strong></div>
                    <div class="ch-stat">SNR: <strong>${ch.snr_est_db} dB</strong></div>
                    <div class="ch-stat">Noise: <strong>${ch.noise_est_dbm} dBm</strong></div>
                </div>
            `;
        }).join("");

        if (mainGrid) mainGrid.innerHTML = html;
        if (fullGrid) fullGrid.innerHTML = html;

        if (snrBarChart) {
            snrBarChart.data.datasets[0].data = snrValues;
            snrBarChart.update("none");
        }
        if (pbusyLineChart) {
            pbusyLineChart.data.datasets[0].data = pbusyValues;
            pbusyLineChart.update("none");
        }
    }

    function renderPendingRequests(pendingList) {
        const tableBody = document.getElementById("pendingRequestsTableBody");
        if (!tableBody) return;

        if (!pendingList || pendingList.length === 0) {
            tableBody.innerHTML = `
                <tr>
                    <td colspan="7" style="text-align: center; color: #10B981; padding: 16px; font-size: 11px;">
                        🟢 <strong>Zero Spectrum Shortfall</strong> — All 4 MNO traffic demands are currently 100% satisfied by AI Dynamic Spectrum Sharing.
                    </td>
                </tr>
            `;
            return;
        }

        tableBody.innerHTML = pendingList.map(req => `
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

    function renderAuditEvents(events) {
        const bodyEl = document.getElementById("fullEventsTableBody");
        if (!bodyEl || !events) return;

        bodyEl.innerHTML = events.map(ev => `
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

    function renderMnoDetailedCards(mnoDetailed) {
        if (!mnoDetailed) return;
        const keys = ["Airtel", "Jio", "Vi", "Bsnl"];
        keys.forEach(key => {
            const data = mnoDetailed[`${key} 6G`];
            if (!data) return;
            setText(`mno${key}Users`, data.users);
            setText(`mno${key}Bw`, data.allocated);
            setText(`mno${key}Demand`, data.demand);
            setText(`mno${key}Bss`, data.bss);
            setText(`mno${key}Sla`, data.sla);
        });
    }

    function renderAIInsights(insights) {
        const container = document.getElementById("aiInsightsContainer");
        if (!container || !insights) return;

        let html = insights.map(item => `
            <div class="insight-item">
                <span class="insight-check">✅</span>
                <div class="insight-text">
                    <p class="i-title">${item.title}</p>
                    <span class="i-badge text-green">${item.badge}</span>
                </div>
            </div>
        `).join("");

        html += `
            <div class="ai-tip-box">
                <span class="tip-icon">💡</span>
                <p>AI is continuously learning from network conditions to optimize spectrum sharing in real-time.</p>
            </div>
        `;
        container.innerHTML = html;
    }

    // -------------------------------------------------------------------------
    // 5. Real-Time Sync Loop (3-second stream)
    // -------------------------------------------------------------------------
    async function fetchDashboardUpdates() {
        try {
            const response = await fetch("/api/dashboard_data");
            if (!response.ok) return;

            const data = await response.json();

            // Top KPIs
            if (data.kpis) {
                setText("kpiTotalSpectrum", data.kpis.total_spectrum);
                setText("kpiUsedSpectrum", data.kpis.used_spectrum);
                setText("kpiUsedPct", `${data.kpis.utilization_pct}%`);
                setText("kpiPendingRequests", data.kpis.pending_requests);
                setText("kpiAvgBss", data.kpis.avg_bss);
                setText("pendingQueueCountHeader", data.kpis.pending_requests);

                const usedBar = document.getElementById("usedProgress");
                if (usedBar) usedBar.style.width = `${data.kpis.utilization_pct}%`;
            }

            // Map and System Status
            if (data.map_info) {
                setText("mapTotalRegions", data.map_info.total_regions);
                setText("mapActiveSectors", data.map_info.active_sectors);
                setText("mapSpectrumPool", data.map_info.spectrum_pool);
            }

            if (data.system_status) {
                setText("statusDrlAgent", data.system_status.drl_agent);
                setText("statusPoolManager", data.system_status.spectrum_pool_manager);
                setText("statusDb", data.system_status.database);
                setText("statusNetSim", data.system_status.network_simulation);
            }

            // AI DRL Engine
            if (data.ai_engine_metrics) {
                setText("aiMseLoss", data.ai_engine_metrics.mse_loss);
                setText("aiPuInterference", data.ai_engine_metrics.pu_interference);
                setText("aiFairnessIndex", data.ai_engine_metrics.fairness_index);
                setText("aiStepBadge", `🟢 Step #${data.step_idx} Active`);

                if (aiRewardChart && data.ai_engine_metrics.reward_series) {
                    aiRewardChart.data.labels = data.ai_engine_metrics.reward_labels;
                    aiRewardChart.data.datasets[0].data = data.ai_engine_metrics.reward_series;
                    aiRewardChart.update("none");
                }
            }

            // Charts & Visualizations
            if (data.analytics_data) {
                if (analyticsCompareChart && data.analytics_data.static_vs_dynamic) {
                    analyticsCompareChart.data.datasets[0].data = data.analytics_data.static_vs_dynamic.static;
                    analyticsCompareChart.data.datasets[1].data = data.analytics_data.static_vs_dynamic.dynamic;
                    analyticsCompareChart.update("none");
                }
                if (fairnessTrendChart && data.analytics_data.fairness_trend) {
                    fairnessTrendChart.data.labels = data.analytics_data.fairness_trend.labels;
                    fairnessTrendChart.data.datasets[0].data = data.analytics_data.fairness_trend.series;
                    fairnessTrendChart.update("none");
                }
            }

            if (utilizationTimeChart && data.utilization_over_time) {
                utilizationTimeChart.data.labels = data.utilization_over_time.labels;
                const series = data.utilization_over_time.series;
                if (series["Airtel 6G"]) utilizationTimeChart.data.datasets[0].data = series["Airtel 6G"];
                if (series["Jio 6G"]) utilizationTimeChart.data.datasets[1].data = series["Jio 6G"];
                if (series["Vi 6G"]) utilizationTimeChart.data.datasets[2].data = series["Vi 6G"];
                if (series["BSNL 6G"]) utilizationTimeChart.data.datasets[3].data = series["BSNL 6G"];
                utilizationTimeChart.update("none");
            }

            if (mnoHistoryChart && data.mno_history_series) {
                mnoHistoryChart.data.labels = data.mno_history_series.labels;
                const hist = data.mno_history_series.series;
                if (hist["Airtel 6G"]) mnoHistoryChart.data.datasets[0].data = hist["Airtel 6G"];
                if (hist["Jio 6G"]) mnoHistoryChart.data.datasets[1].data = hist["Jio 6G"];
                if (hist["Vi 6G"]) mnoHistoryChart.data.datasets[2].data = hist["Vi 6G"];
                if (hist["BSNL 6G"]) mnoHistoryChart.data.datasets[3].data = hist["BSNL 6G"];
                mnoHistoryChart.update("none");
            }

            if (spectrumDistributionChart && data.spectrum_distribution) {
                spectrumDistributionChart.data.datasets[0].data = data.spectrum_distribution.map(d => d.mhz);
                spectrumDistributionChart.update("none");
            }

            if (bssComparisonChart && data.bss_comparison) {
                bssComparisonChart.data.datasets[0].data = data.bss_comparison.map(d => d.score);
                bssComparisonChart.update("none");
            }

            if (requestAllocationChart && data.requests_vs_allocation) {
                requestAllocationChart.data.datasets[0].data = data.requests_vs_allocation.requests;
                requestAllocationChart.data.datasets[1].data = data.requests_vs_allocation.allocated;
                requestAllocationChart.update("none");
            }

            // Tables & Lists
            renderMnoDetailedCards(data.mno_detailed);
            renderPendingRequests(data.pending_requests_list);
            renderAuditEvents(data.audit_events_list);
            renderAIInsights(data.ai_insights);
            renderChannelGrid(data.channels);

            // MNO summary table
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

            // Recent AI decisions table
            const decisionsBody = document.getElementById("aiDecisionsBody");
            if (decisionsBody && data.recent_ai_decisions) {
                decisionsBody.innerHTML = data.recent_ai_decisions.map(d => `
                    <tr>
                        <td>${d.time}</td>
                        <td>${d.event}</td>
                        <td>${d.action}</td>
                    </tr>
                `).join("");
            }

            // Live events feed
            const eventsList = document.getElementById("eventsFeedList");
            if (eventsList && data.live_events) {
                eventsList.innerHTML = data.live_events.map(ev => `
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

        } catch (err) {
            console.error("Dashboard synchronization error:", err);
        }
    }

    // Refresh every 3 seconds
    setInterval(fetchDashboardUpdates, 3000);
    fetchDashboardUpdates();
});
