<script lang="ts">
	import { pipelineStore, type VideoInfo } from '$lib/stores/pipeline.svelte';
	import PipelineWizard from '$lib/components/PipelineWizard.svelte';
	import ProgressBar from '$lib/components/ProgressBar.svelte';
	import StatusBadge from '$lib/components/StatusBadge.svelte';

	// Wizard steps
	const wizardSteps = [
		{ label: 'Загрузка', description: 'Загрузка видеозаписи SCADA' },
		{ label: 'Обработка', description: 'VLM конвейер обработки' },
		{ label: 'Результаты', description: 'XML и отправка' },
	];

	let dragOver = $state(false);
	let uploadStatus = $state<'idle' | 'uploading' | 'success' | 'error'>('idle');
	let selectedFile = $state<File | null>(null);
	let errorMessage = $state('');

	// Parameter table state
	let paramTableFile = $state<File | null>(null);
	let paramTableStatus = $state<'idle' | 'uploading' | 'success' | 'error'>('idle');
	let paramTableMessage = $state('');

	// Derive uploadedVideo and wizardStep from store (persists across navigation)
	let uploadedVideo = $derived(pipelineStore.uploadedVideo);
	let wizardStep = $derived.by(() => {
		if (pipelineStore.pipelineStatus === 'completed') return 2;
		if (pipelineStore.isProcessing) return 1;
		if (pipelineStore.uploadedVideo) return 1;
		return 0;
	});

	// Check VLM health when video is uploaded
	$effect(() => {
		if (pipelineStore.uploadedVideo) {
			pipelineStore.checkVlmHealth();
		}
	});

	function handleDragOver(e: DragEvent) { e.preventDefault(); dragOver = true; }
	function handleDragLeave() { dragOver = false; }
	function handleDrop(e: DragEvent) {
		e.preventDefault();
		dragOver = false;
		const files = e.dataTransfer?.files;
		if (files && files.length > 0) selectedFile = files[0];
	}
	function handleFileSelect(e: Event) {
		const input = e.target as HTMLInputElement;
		if (input.files && input.files.length > 0) selectedFile = input.files[0];
	}

	async function handleUpload() {
		if (!selectedFile) return;
		uploadStatus = 'uploading';
		errorMessage = '';
		const result = await pipelineStore.uploadVideo(selectedFile);
		if (result) {
			uploadStatus = 'success';
		} else {
			uploadStatus = 'error';
			errorMessage = pipelineStore.lastError;
		}
	}

	function handleParamTableSelect(e: Event) {
		const input = e.target as HTMLInputElement;
		if (input.files && input.files.length > 0) paramTableFile = input.files[0];
	}

	async function handleParamTableUpload() {
		if (!paramTableFile) return;
		paramTableStatus = 'uploading';
		paramTableMessage = '';
		const videoId = uploadedVideo?.video_id;
		const result = await pipelineStore.uploadParameterTable(paramTableFile, videoId);
		if (result) {
			paramTableStatus = 'success';
			paramTableMessage = result.message;
		} else {
			paramTableStatus = 'error';
			paramTableMessage = pipelineStore.lastError;
		}
	}

	async function handleStartPipeline() {
		if (!uploadedVideo) return;
		const success = await pipelineStore.startPipeline(uploadedVideo.video_id);
		if (!success) {
			errorMessage = pipelineStore.lastError;
		}
	}

	function handleNewVideo() {
		selectedFile = null;
		uploadStatus = 'idle';
		errorMessage = '';
		paramTableFile = null;
		paramTableStatus = 'idle';
		paramTableMessage = '';
		pipelineStore.resetPipeline();
	}

	let videoTypeLabel = $derived(
		uploadedVideo?.resolution?.includes('1912') ? 'Прямая съёмка' :
		uploadedVideo?.resolution?.includes('1280') ? 'Ручная камера' : 'Автоопределение'
	);
</script>

<div class="space-y-6">
	<!-- Header -->
	<div class="flex items-center justify-between animate-fade-in">
		<div>
			<h2 class="text-2xl font-bold text-text">VLM Конвейер</h2>
			<p class="text-sm text-text-secondary mt-1">Загрузка и обработка видеозаписей SCADA с помощью VLM</p>
		</div>
		<div class="flex items-center gap-3">
			<StatusBadge variant={pipelineStore.vlmHealthy ? 'success' : 'error'} pulse={!pipelineStore.vlmHealthy}>
				VLM: {pipelineStore.vlmHealthy ? 'Доступен' : 'Недоступен'}
			</StatusBadge>
			<StatusBadge variant={pipelineStore.wsConnected ? 'success' : 'error'}>
				{pipelineStore.wsConnected ? 'Подключено' : 'Оффлайн'}
			</StatusBadge>
		</div>
	</div>

	<!-- Wizard steps -->
	<PipelineWizard currentStep={wizardStep} steps={wizardSteps} />

	<!-- Step 0: Upload video -->
	{#if wizardStep >= 0}
		<div class="bg-elevated border border-border rounded-xl p-6 animate-fade-in">
			<h3 class="text-lg font-semibold text-text mb-4">Загрузка видеозаписи</h3>
			<div class="grid grid-cols-1 lg:grid-cols-[1fr_300px] gap-6">
				<div>
					<div
						class="border-2 border-dashed rounded-xl p-10 text-center transition-all duration-300 cursor-pointer
							{dragOver ? 'border-accent bg-accent/5 scale-[1.01]' : 'border-border hover:border-accent/50 hover:bg-overlay/50'}"
						ondragover={handleDragOver}
						ondragleave={handleDragLeave}
						ondrop={handleDrop}
						onclick={() => document.getElementById('file-input')?.click()}
						role="button"
						tabindex="0"
						onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') document.getElementById('file-input')?.click() }}
					>
						<input id="file-input" type="file" accept="video/*" class="hidden" onchange={handleFileSelect} />
						<div class="space-y-3">
							<svg class="w-12 h-12 text-text-secondary mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
								<path stroke-linecap="round" stroke-linejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
							</svg>
							{#if selectedFile}
								<div>
									<p class="text-accent font-mono text-sm">{selectedFile.name}</p>
									<p class="text-text-secondary text-xs mt-1">{(selectedFile.size / (1024 * 1024)).toFixed(1)} МБ</p>
								</div>
							{:else}
								<div>
									<p class="text-text font-medium">Перетащите видеофайл сюда</p>
									<p class="text-text-secondary text-sm mt-1">или нажмите для выбора</p>
								</div>
							{/if}
							<p class="text-xs text-text-secondary">Поддерживается: MP4, AVI, MKV, MOV</p>
						</div>
					</div>

					<div class="mt-4 flex gap-3">
						<button
							onclick={handleUpload}
							disabled={!selectedFile || uploadStatus === 'uploading'}
							class="flex-1 px-6 py-3 bg-accent text-base rounded-lg font-semibold text-sm
								transition-all duration-200 hover:bg-accent-hover hover:shadow-lg hover:shadow-accent/20
								active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed"
						>
							{#if uploadStatus === 'uploading'}Загрузка...{:else}Загрузить видео{/if}
						</button>
					</div>
				</div>

				<!-- Video info -->
				{#if uploadedVideo}
					<div class="bg-overlay border border-border rounded-xl p-4 animate-slide-up">
						<h4 class="text-sm font-semibold text-text mb-3">Информация о видео</h4>
						<div class="space-y-2 font-mono text-sm">
							<div class="flex justify-between"><span class="text-text-secondary">ID:</span><span class="text-accent">{uploadedVideo.video_id.slice(0, 8)}...</span></div>
							<div class="flex justify-between"><span class="text-text-secondary">Тип:</span><StatusBadge variant="info">{videoTypeLabel}</StatusBadge></div>
							<div class="flex justify-between"><span class="text-text-secondary">Разрешение:</span><span class="text-text">{uploadedVideo.resolution}</span></div>
							<div class="flex justify-between"><span class="text-text-secondary">FPS:</span><span class="text-text">{uploadedVideo.fps.toFixed(1)}</span></div>
							<div class="flex justify-between"><span class="text-text-secondary">Длительность:</span><span class="text-text">{Math.floor(uploadedVideo.duration_s / 60)}:{Math.floor(uploadedVideo.duration_s % 60).toString().padStart(2, '0')}</span></div>
							<div class="flex justify-between"><span class="text-text-secondary">Кадры:</span><span class="text-text">{uploadedVideo.total_frames}</span></div>
						</div>
					</div>
				{/if}
			</div>
		</div>
	{/if}

	<!-- Step 1: Processing -->
	{#if wizardStep >= 1 && uploadedVideo}
		<!-- Parameter Table Upload -->
		<div class="bg-elevated border border-border rounded-xl p-6 animate-fade-in mb-4">
			<div class="flex items-center justify-between mb-4">
				<div>
					<h3 class="text-lg font-semibold text-text">Таблица параметров</h3>
					<p class="text-sm text-text-secondary mt-1">Загрузите xlsx/csv таблицу для точного сопоставления параметров мнемосхемы</p>
				</div>
				<StatusBadge variant={pipelineStore.parameterTableLoaded ? 'success' : 'info'}>
					{pipelineStore.parameterTableLoaded ? `${pipelineStore.parameterTableCount} параметров` : 'Не загружена'}
				</StatusBadge>
			</div>

			<div class="flex gap-3 items-center">
				<input
					id="param-table-input"
					type="file"
					accept=".xlsx,.csv"
					class="block w-full text-sm text-text-secondary
						file:mr-4 file:py-2 file:px-4
						file:rounded-lg file:border-0
						file:text-sm file:font-semibold
						file:bg-accent/10 file:text-accent
						hover:file:bg-accent/20 file:cursor-pointer
						file:transition-colors"
					onchange={handleParamTableSelect}
				/>
				<button
					onclick={handleParamTableUpload}
					disabled={!paramTableFile || paramTableStatus === 'uploading'}
					class="px-4 py-2 bg-accent text-base rounded-lg font-semibold text-sm
						transition-all duration-200 hover:bg-accent-hover hover:shadow-lg hover:shadow-accent/20
						active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap"
				>
					{#if paramTableStatus === 'uploading'}Загрузка...{:else}Загрузить таблицу{/if}
				</button>
			</div>

			{#if paramTableStatus === 'success' && paramTableMessage}
				<div class="mt-3 flex items-center gap-2 text-sm">
					<svg class="w-4 h-4 text-success" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
						<path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7" />
					</svg>
					<span class="text-success">{paramTableMessage}</span>
				</div>
			{/if}
			{#if paramTableStatus === 'error' && paramTableMessage}
				<div class="mt-3 text-sm text-danger">{paramTableMessage}</div>
			{/if}
		</div>

		<div class="bg-elevated border border-border rounded-xl p-6 animate-fade-in">
			<h3 class="text-lg font-semibold text-text mb-2">Обработка VLM</h3>
			<p class="text-sm text-text-secondary mb-4">Видеозапись обрабатывается с помощью локального VLM сервера для извлечения параметров SCADA</p>

			<div class="grid grid-cols-1 lg:grid-cols-[1fr_350px] gap-6">
				<div class="space-y-4">
					<!-- Processing status -->
					{#if pipelineStore.isProcessing}
						<div class="bg-overlay border border-border rounded-lg p-4">
							<div class="flex items-center justify-between mb-3">
								<span class="text-sm text-text-secondary font-mono">Статус</span>
								<StatusBadge variant="warning" pulse>Обработка</StatusBadge>
							</div>
							<ProgressBar 
								value={pipelineStore.progress} 
								label={pipelineStore.stepLabel || 'Обработка...'}
								variant="default" 
								size="lg" 
							/>
							<div class="mt-3 text-xs text-text-secondary font-mono space-y-1">
								<div>Кадры: {pipelineStore.framesProcessed} / {pipelineStore.totalFrames || '?'}</div>
							</div>
						</div>
					{:else if pipelineStore.pipelineStatus === 'completed'}
						<div class="bg-overlay border border-success/30 rounded-lg p-4 animate-slide-up">
							<div class="flex items-center gap-2 mb-2">
								<svg class="w-5 h-5 text-success" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
									<path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7" />
								</svg>
								<span class="text-success font-semibold">Обработка завершена</span>
							</div>
							<p class="text-sm text-text-secondary">XML-отчёт готов к просмотру и отправке</p>
						</div>
					{:else}
						<div class="bg-overlay border border-border rounded-lg p-4">
							<div class="flex items-center gap-3 mb-3">
								<div class="h-8 w-8 rounded-lg bg-accent/10 border border-accent/20 flex items-center justify-center">
									<svg class="w-4 h-4 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
										<path stroke-linecap="round" stroke-linejoin="round" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
										<path stroke-linecap="round" stroke-linejoin="round" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
									</svg>
								</div>
								<div>
									<div class="text-sm font-semibold text-text">Готово к обработке</div>
									<div class="text-xs text-text-secondary">VLM сервер: {pipelineStore.vlmHealthy ? 'доступен' : 'недоступен'}</div>
								</div>
							</div>
							<button 
								onclick={handleStartPipeline} 
								disabled={!pipelineStore.vlmHealthy || pipelineStore.isProcessing}
								class="w-full px-6 py-3 bg-accent text-base rounded-lg font-semibold text-sm
									transition-all duration-200 hover:bg-accent-hover hover:shadow-lg hover:shadow-accent/20
									active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed"
							>
								{#if pipelineStore.isProcessing}Обработка...{:else}Запустить VLM обработку{/if}
							</button>
						</div>
					{/if}
				</div>

				<!-- VLM Pipeline Steps -->
				<div class="bg-overlay border border-border rounded-lg p-4">
					<h4 class="text-sm font-semibold text-text mb-3">Этапы обработки</h4>
					<div class="space-y-3">
						{#each [
							{ step: '1', label: 'Извлечение кадров', desc: 'Каждые 500мс из видео' },
							{ step: '2', label: 'VLM анализ', desc: 'Мнемосхема → параметры' },
							{ step: '3', label: 'Генерация XML', desc: 'Формат <sheme>' },
							{ step: '4', label: 'GPG шифрование', desc: 'AES256 + RSA 4096' },
						] as item, i}
							<div class="flex gap-3 animate-slide-up" style="animation-delay: {i * 80}ms">
								<div class="shrink-0 w-6 h-6 rounded-full bg-accent/20 text-accent flex items-center justify-center text-xs font-mono font-bold">{item.step}</div>
								<div><div class="text-xs text-text font-medium">{item.label}</div><div class="text-[10px] text-text-secondary mt-0.5">{item.desc}</div></div>
							</div>
						{/each}
					</div>
				</div>
			</div>
		</div>
	{/if}

	<!-- Step 2: Results -->
	{#if wizardStep >= 2 && pipelineStore.pipelineStatus === 'completed'}
		<div class="bg-elevated border border-border rounded-xl p-6 animate-fade-in">
			<h3 class="text-lg font-semibold text-text mb-4">Результаты</h3>
			
			<div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
				<div class="bg-overlay border border-success/30 rounded-lg p-4 text-center">
					<div class="text-2xl font-mono font-bold text-success">✓</div>
					<div class="text-xs text-text-secondary font-mono mt-1">Обработка завершена</div>
				</div>
				<div class="bg-overlay border border-border rounded-lg p-4 text-center">
					<div class="text-2xl font-mono font-bold text-accent">{pipelineStore.totalFrames || '?'}</div>
					<div class="text-xs text-text-secondary font-mono mt-1">Кадров обработано</div>
				</div>
				<div class="bg-overlay border border-border rounded-lg p-4 text-center">
					<div class="text-2xl font-mono font-bold text-text">{pipelineStore.activeVideoId.slice(0, 8)}</div>
					<div class="text-xs text-text-secondary font-mono mt-1">ID сессии</div>
				</div>
			</div>

			<div class="flex gap-3">
				<a
					href="/results"
					class="flex-1 px-6 py-3 bg-accent text-base rounded-lg font-semibold text-sm text-center
						transition-all duration-200 hover:bg-accent-hover hover:shadow-lg hover:shadow-accent/20"
				>
					Посмотреть XML и отправить
				</a>
				<button
					onclick={handleNewVideo}
					class="px-6 py-3 bg-elevated border border-border text-text-secondary rounded-lg font-semibold text-sm
						transition-all duration-200 hover:text-text hover:border-accent/50"
				>
					Новое видео
				</button>
			</div>
		</div>
	{/if}

	<!-- Error -->
	{#if errorMessage}
		<div class="bg-danger/10 border border-danger/30 rounded-lg p-4 animate-fade-in">
			<div class="flex items-center gap-2">
				<span class="text-danger font-bold">Ошибка:</span>
				<span class="text-danger/80 font-mono text-sm">{errorMessage}</span>
			</div>
		</div>
	{/if}
</div>
