import Vue from 'vue'
import Vuex from 'vuex'
import createPersistedState from 'vuex-persistedstate'
import patient from './patients/patient'
import medication from './medications/medication'
import observation from './observations/observation'
import statement from './medication_statements/statement'

Vue.use(Vuex)

export default new Vuex.Store({
  modules: {
    patient,
    medication,
    observation,
    statement
  },
  plugins: [
    createPersistedState()
  ]
})
