<script lang="ts">
	import { authStore } from '$lib/stores/auth.svelte';
	import { goto } from '$app/navigation';

	let username = $state('');
	let email = $state('');
	let password = $state('');
	let confirmPassword = $state('');
	let fullName = $state('');
	let showPassword = $state(false);

	let passwordMismatch = $derived(confirmPassword.length > 0 && password !== confirmPassword);

	async function handleRegister() {
		if (passwordMismatch) return;
		const success = await authStore.register(username, email, password, fullName);
		if (success) {
			goto('/pipeline');
		}
	}

	function handleKeyDown(e: KeyboardEvent) {
		if (e.key === 'Enter' && username && email && password && !passwordMismatch) {
			handleRegister();
		}
	}
</script>

<div class="min-h-screen flex items-center justify-center bg-base p-4 relative overflow-hidden">
	<!-- Background decoration -->
	<div class="absolute inset-0 pointer-events-none">
		<div class="absolute top-1/3 left-1/3 w-[500px] h-[500px] bg-accent/3 rounded-full blur-[120px]"></div>
		<div class="absolute bottom-1/3 right-1/3 w-[400px] h-[400px] bg-info/3 rounded-full blur-[120px]"></div>
	</div>

	<div class="relative w-full max-w-sm animate-scale-in">
		<div class="text-center mb-8">
			<div class="inline-flex items-center gap-2.5 mb-3">
				<div class="h-9 w-9 rounded-lg bg-accent/10 border border-accent/25 flex items-center justify-center">
					<svg class="w-5 h-5 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
						<path stroke-linecap="round" stroke-linejoin="round" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
					</svg>
				</div>
				<h1 class="text-xl font-semibold text-text">InfoDiode</h1>
			</div>
			<p class="text-text-tertiary text-sm">Создайте аккаунт для начала работы</p>
		</div>

		<div class="bg-elevated border border-border rounded-xl p-6 shadow-2xl shadow-black/30">
			{#if authStore.error}
				<div class="mb-4 px-3 py-2.5 bg-danger/10 border border-danger/20 rounded-lg animate-fade-in">
					<p class="text-sm text-danger">{authStore.error}</p>
				</div>
			{/if}

			<div class="space-y-4">
				<div>
					<label for="fullName" class="block text-xs font-medium text-text-secondary mb-1.5">ФИО</label>
					<input
						id="fullName"
						type="text"
						bind:value={fullName}
						onkeydown={handleKeyDown}
						placeholder="Иванов Иван Иванович"
						class="w-full px-3 py-2.5 bg-base border border-border rounded-lg text-text text-sm
							placeholder:text-text-tertiary/50
							focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30
							transition-colors duration-200"
					/>
				</div>

				<div>
					<label for="username" class="block text-xs font-medium text-text-secondary mb-1.5">Логин</label>
					<input
						id="username"
						type="text"
						bind:value={username}
						onkeydown={handleKeyDown}
						autocomplete="username"
						placeholder="Введите логин"
						class="w-full px-3 py-2.5 bg-base border border-border rounded-lg text-text font-mono text-sm
							placeholder:text-text-tertiary/50
							focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30
							transition-colors duration-200"
					/>
				</div>

				<div>
					<label for="email" class="block text-xs font-medium text-text-secondary mb-1.5">Email</label>
					<input
						id="email"
						type="email"
						bind:value={email}
						onkeydown={handleKeyDown}
						autocomplete="email"
						placeholder="user@example.com"
						class="w-full px-3 py-2.5 bg-base border border-border rounded-lg text-text font-mono text-sm
							placeholder:text-text-tertiary/50
							focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30
							transition-colors duration-200"
					/>
				</div>

				<div>
					<label for="password" class="block text-xs font-medium text-text-secondary mb-1.5">Пароль</label>
					<div class="relative">
						<input
							id="password"
							type={showPassword ? 'text' : 'password'}
							bind:value={password}
							onkeydown={handleKeyDown}
							autocomplete="new-password"
							placeholder="Минимум 4 символа"
							class="w-full px-3 py-2.5 bg-base border border-border rounded-lg text-text font-mono text-sm
								placeholder:text-text-tertiary/50
								focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30
								transition-colors duration-200 pr-9"
						/>
						<button
							type="button"
							onclick={() => showPassword = !showPassword}
							class="absolute right-2.5 top-1/2 -translate-y-1/2 text-text-tertiary hover:text-text transition-colors"
						>
							<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
								{#if showPassword}
									<path stroke-linecap="round" stroke-linejoin="round" d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
								{:else}
									<path stroke-linecap="round" stroke-linejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
									<path stroke-linecap="round" stroke-linejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
								{/if}
							</svg>
						</button>
					</div>
				</div>

				<div>
					<label for="confirmPassword" class="block text-xs font-medium text-text-secondary mb-1.5">Подтверждение пароля</label>
					<input
						id="confirmPassword"
						type="password"
						bind:value={confirmPassword}
						onkeydown={handleKeyDown}
						autocomplete="new-password"
						placeholder="Повторите пароль"
						class="w-full px-3 py-2.5 bg-base border border-border rounded-lg text-text font-mono text-sm
							placeholder:text-text-tertiary/50
							focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30
							transition-colors duration-200
							{passwordMismatch ? 'border-danger focus:border-danger focus:ring-danger/30' : ''}"
					/>
					{#if passwordMismatch}
						<p class="text-xs text-danger mt-1">Пароли не совпадают</p>
					{/if}
				</div>

				<button
					onclick={handleRegister}
					disabled={!username || !email || !password || passwordMismatch || authStore.loading}
					class="w-full px-4 py-2.5 bg-accent text-white rounded-lg font-medium text-sm
						transition-all duration-200
						hover:bg-accent-hover
						active:scale-[0.98]
						disabled:opacity-40 disabled:cursor-not-allowed"
				>
					{#if authStore.loading}
						Регистрация...
					{:else}
						Зарегистрироваться
					{/if}
				</button>
			</div>

			<div class="mt-5 text-center">
				<a href="/auth/login" class="text-sm text-accent hover:text-accent-hover transition-colors duration-200">
					Уже есть аккаунт? Войти
				</a>
			</div>
		</div>
	</div>
</div>
