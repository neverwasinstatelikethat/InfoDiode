<script lang="ts">
	import { onMount } from 'svelte';
	import { pipelineStore, type PipelineStatusResponse } from '$lib/stores/pipeline.svelte';
	import { authStore } from '$lib/stores/auth.svelte';
	import { API_BASE } from '$lib/config';
	import StatusBadge from '$lib/components/StatusBadge.svelte';

	/** QR status for a video (matches backend /qr/status/{video_id} response) */
	interface QrStatus {
		video_id: string;
		overlay_exists: boolean;
		overlay_path?: string;
		overlay_size_bytes?: number;
		overlay_created_at?: string;
		source_xml_exists: boolean;
		source_video_exists: boolean;
		can_generate: boolean;
	}

	/** Extended video info with QR status */
	interface VideoWithQrStatus {
		video_id: string;
		status: PipelineStatusResponse['status'];
		progress_pct: number;
		qrStatus: QrStatus | null;
		qrLoading: boolean;
		generating: boolean;
	}

	let videos = $state<VideoWithQrStatus[]>([]);
	let loading = $state(true);
	let errorMessage = $state('');
	let refreshInterval: ReturnType<typeof setInterval> | null = null;

	/** Fetch QR status for a single video */
	async function fetchQrStatus(videoId: string): Promise<QrStatus | null> {
		try {
			const res = await fetch(`${API_BASE}/qr/status/${videoId}`, {
				headers: authStore.authHeaders(),
			});
			if (res.ok) {
				return await res.json() as QrStatus;
			}
			return null;
		} catch {
			return null;
		}
	}

	/** Generate QR overlay video */
	async function generateQrVideo(videoId: string) {
		const video = videos.find(v => v.video_id === videoId);
		if (!video || video.generating) return;

		video.generating = true;
		errorMessage = '';

		try {
			const res = await fetch(`${API_BASE}/qr/generate/${videoId}`, {
				method: 'POST',
				headers: authStore.authHeaders(),
			});
			if (!res.ok) {
				const data = await res.json().catch(() => ({}));
				errorMessage = data.detail || `Ошибка генерации: ${res.statusText}`;
			}
			// Status will be updated via polling
		} catch (e) {
			errorMessage = `Ошибка генерации: ${e}`;
		}
	}

	/** Download QR overlay video */
	function downloadQrVideo(videoId: string) {
		const link = document.createElement('a');
		link.href = `${API_BASE}/qr/video/${videoId}`;
		link.download = `qr_overlay_${videoId.slice(0, 8)}.mp4`;
		document.body.appendChild(link);
		link.click();
		document.body.removeChild(link);
	}

	/** Load all completed videos and their QR status */
	async function loadVideos() {
		try {
			const res = await fetch(`${API_BASE}/pipeline/completed`, {
				headers: authStore.authHeaders(),
			});
			if (!res.ok) {
				errorMessage = `Ошибка загрузки: ${res.statusText}`;
				return;
			}

			const pipelineStatuses: PipelineStatusResponse[] = await res.json();

			// Map to our format
			const completedVideos = pipelineStatuses.map((p: PipelineStatusResponse): VideoWithQrStatus => {
				const existing = videos.find(v => v.video_id === p.video_id);
				return {
					video_id: p.video_id,
					status: p.status,
					progress_pct: p.progress_pct,
					qrStatus: existing?.qrStatus || null,
					qrLoading: existing?.qrLoading ?? true,
					generating: existing?.generating ?? false,
				};
			});

			videos = completedVideos;

			// Fetch QR status for videos that need it
			for (const video of videos) {
				if (!video.qrStatus && video.qrLoading) {
					video.qrStatus = await fetchQrStatus(video.video_id);
					video.qrLoading = false;
				}
			}
		} catch (e) {
			errorMessage = `Ошибка загрузки видео: ${e}`;
		} finally {
			loading = false;
		}
	}

	/** Refresh QR statuses periodically */
	async function refreshQrStatuses() {
		for (const video of videos) {
			if (video.generating || !video.qrStatus?.overlay_exists) {
				const newStatus = await fetchQrStatus(video.video_id);
				if (newStatus) {
					video.qrStatus = newStatus;
					if (newStatus.overlay_exists) {
						video.generating = false;
					}
				}
			}
		}
	}

	onMount(() => {
		loadVideos();
		// Poll for QR status updates every 3 seconds
		refreshInterval = setInterval(refreshQrStatuses, 3000);
		return () => {
			if (refreshInterval) clearInterval(refreshInterval);
		};
	});

	/** Truncate video ID for display */
	function truncateId(id: string): string {
		return `${id.slice(0, 8)}...${id.slice(-4)}`;
	}
</script>

<div class="space-y-6">
	<!-- Header -->
	<div class="flex items-center justify-between animate-fade-in">
		<div>
			<h2 class="text-2xl font-bold text-text">QR Инфодиод</h2>
			<p class="text-sm text-text-secondary mt-1">Генерация видео с QR-кодами для air-gapped передачи данных</p>
		</div>
		<button
			onclick={loadVideos}
			disabled={loading}
			class="px-4 py-2 bg-elevated border border-border text-text-secondary rounded-lg text-sm font-medium
				transition-all duration-200 hover:text-text hover:border-accent/50 disabled:opacity-50"
		>
			{#if loading}
				<span class="flex items-center gap-2">
					<svg class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
						<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
						<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
					</svg>
					Обновление...
				</span>
			{:else}
				Обновить список
			{/if}
		</button>
	</div>

	<!-- Info card -->
	<div class="bg-elevated border border-border rounded-xl p-6 animate-fade-in">
		<div class="flex items-start gap-4">
			<div class="h-10 w-10 rounded-lg bg-accent/10 border border-accent/20 flex items-center justify-center shrink-0">
				<svg class="w-5 h-5 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
					<path stroke-linecap="round" stroke-linejoin="round" d="M12 4v1m6 11h2m-6 0h-2v4m0-11v3m0 0h.01M12 12h4.01M16 20h4M4 12h4m12 0h.01M5 8h2a1 1 0 001-1V5a1 1 0 00-1-1H5a1 1 0 00-1 1v2a1 1 0 001 1zm12 0h2a1 1 0 001-1V5a1 1 0 00-1-1h-2a1 1 0 00-1 1v2a1 1 0 001 1zM5 20h2a1 1 0 001-1v-2a1 1 0 00-1-1H5a1 1 0 00-1 1v2a1 1 0 001 1z" />
				</svg>
			</div>
			<div>
				<h3 class="text-sm font-semibold text-text mb-1">Как работает QR инфодиод</h3>
				<p class="text-sm text-text-secondary leading-relaxed">
					Система генерирует видео с наложенными QR-кодами, содержащими сжатые данные XML.
					Используется формат QR v40 с максимальной плотностью данных, msgpack + zlib сжатие.
					Готовое видео можно воспроизвести на air-gapped системе и сканировать QR-коды для извлечения данных.
				</p>
			</div>
		</div>
	</div>

	<!-- Video list -->
	<div class="bg-elevated border border-border rounded-xl overflow-hidden animate-fade-in">
		<div class="px-6 py-4 border-b border-border">
			<h3 class="text-lg font-semibold text-text">Обработанные видео</h3>
		</div>

		{#if loading && videos.length === 0}
			<div class="p-12 text-center">
				<svg class="w-8 h-8 text-text-secondary animate-spin mx-auto mb-3" fill="none" viewBox="0 0 24 24">
					<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
					<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
				</svg>
				<p class="text-text-secondary text-sm">Загрузка списка видео...</p>
			</div>
		{:else if videos.length === 0}
			<div class="p-12 text-center">
				<div class="h-12 w-12 rounded-full bg-overlay border border-border flex items-center justify-center mx-auto mb-3">
					<svg class="w-6 h-6 text-text-secondary" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
						<path stroke-linecap="round" stroke-linejoin="round" d="M7 4v16M17 4v16M3 8h4m10 0h4M3 12h18M3 16h4m10 0h4M4 20h16a1 1 0 001-1V5a1 1 0 00-1-1H4a1 1 0 00-1 1v14a1 1 0 001 1z" />
					</svg>
				</div>
				<p class="text-text font-medium mb-1">Нет обработанных видео</p>
				<p class="text-text-secondary text-sm">Завершите обработку видео в разделе "Пайплайн"</p>
				<a
					href="/pipeline"
					class="inline-flex items-center gap-2 mt-4 px-4 py-2 bg-accent text-white rounded-lg text-sm font-medium
						transition-all duration-200 hover:bg-accent-hover"
				>
					<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
						<path stroke-linecap="round" stroke-linejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
					</svg>
					Перейти к пайплайну
				</a>
			</div>
		{:else}
			<div class="divide-y divide-border">
				{#each videos as video (video.video_id)}
					<div class="p-6 flex items-center justify-between gap-4 hover:bg-overlay/50 transition-colors duration-200">
						<div class="flex items-center gap-4">
							<div class="h-10 w-10 rounded-lg bg-accent/10 border border-accent/20 flex items-center justify-center shrink-0">
								<svg class="w-5 h-5 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
									<path stroke-linecap="round" stroke-linejoin="round" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
								</svg>
							</div>
							<div>
								<div class="flex items-center gap-2">
									<span class="font-mono text-sm text-text font-medium">{truncateId(video.video_id)}</span>
									<button
										onclick={() => navigator.clipboard.writeText(video.video_id)}
										class="text-text-tertiary hover:text-accent transition-colors"
										title="Копировать ID"
									>
										<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
											<path stroke-linecap="round" stroke-linejoin="round" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
										</svg>
									</button>
								</div>
								<div class="flex items-center gap-2 mt-1">
									{#if video.qrLoading}
										<span class="text-xs text-text-secondary">Проверка статуса...</span>
									{:else if video.generating}
										<StatusBadge variant="warning" pulse>Генерация...</StatusBadge>
									{:else if video.qrStatus?.overlay_exists}
										<StatusBadge variant="success">QR видео готово</StatusBadge>
									{:else}
										<StatusBadge variant="info">Ожидает генерации</StatusBadge>
									{/if}
								</div>
							</div>
						</div>

						<div class="flex items-center gap-2">
							{#if video.qrStatus?.overlay_exists}
								<button
									onclick={() => downloadQrVideo(video.video_id)}
									class="flex items-center gap-2 px-4 py-2 bg-success/10 border border-success/30 text-success rounded-lg text-sm font-medium
										transition-all duration-200 hover:bg-success/20"
								>
									<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
										<path stroke-linecap="round" stroke-linejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
									</svg>
									Скачать
								</button>
							{:else}
								<button
									onclick={() => generateQrVideo(video.video_id)}
									disabled={video.generating || !video.qrStatus?.can_generate}
									class="flex items-center gap-2 px-4 py-2 bg-accent text-white rounded-lg text-sm font-medium
										transition-all duration-200 hover:bg-accent-hover hover:shadow-lg hover:shadow-accent/20
										active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed"
								>
									{#if video.generating}
										<svg class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
											<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
											<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
										</svg>
										Генерация...
									{:else}
										<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
											<path stroke-linecap="round" stroke-linejoin="round" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
											<path stroke-linecap="round" stroke-linejoin="round" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
										</svg>
										Сгенерировать QR видео
									{/if}
								</button>
							{/if}
						</div>
					</div>
				{/each}
			</div>
		{/if}
	</div>

	<!-- Error message -->
	{#if errorMessage}
		<div class="bg-danger/10 border border-danger/30 rounded-lg p-4 animate-fade-in">
			<div class="flex items-center gap-2">
				<svg class="w-5 h-5 text-danger" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
					<path stroke-linecap="round" stroke-linejoin="round" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
				</svg>
				<span class="text-danger font-medium">{errorMessage}</span>
			</div>
		</div>
	{/if}
</div>
