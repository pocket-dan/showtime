<template>
  <div class="text">
    <router-link to="/">
      <img id="logo" src="~@/assets/logo2.png" alt="electron-vue">
    </router-link>
  </div>
</template>

<script>

</script>

<style scoped>
  #logo {
    height: 69px;
    width: auto;
  }

  .text{
    position: relative;
    display: inline-block;
    font-size: 2em;
  }

  .text:before{
    position: absolute;
    top: 2.3em;
    left: 0;
    content: "";
    display: inline-block;
    width: 0;
    height: 2px;
    background: #41ADAE;
    transition: 2s;
  }

  .text:hover:before{
    width: 100%;
  }
</style>
